/********************************************************************************************************************************************
 * Saras
 * 
 * Copyright (C) 2019, Mahendra K. Verma
 *
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     1. Redistributions of source code must retain the above copyright
 *        notice, this list of conditions and the following disclaimer.
 *     2. Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimer in the
 *        documentation and/or other materials provided with the distribution.
 *     3. Neither the name of the copyright holder nor the
 *        names of its contributors may be used to endorse or promote products
 *        derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ********************************************************************************************************************************************
 */
/*! \file poisson2.cc
 *
 *  \brief Definitions for functions of class poisson for 2D
 *  \sa poisson.h
 *  \author Roshan Samuel
 *  \date Nov 2019
 *  \copyright New BSD License
 *
 ********************************************************************************************************************************************
 */

#include "poisson.h"

/**
 ********************************************************************************************************************************************
 * \brief   Constructor of the multigrid_d2 class derived from the poisson class
 *
 *          The constructor of the derived multigrid_d2 class frst calls the base poisson class with the arguments passed to it.
 *          Most of the array initializations are performed in the constructor of the base class.
 *          Initializations unique to 2D implementation of multi-grid solver are then done here.
 *          Primarily this involves generating the MPI datatypes for data transfer between sub-domain boundaries.
 *          When testing the Poisson solver with Dirichlet BC, the analytical 2D solution for the test is also calculated here
 *          through a call to initDirichlet() function.
 *
 * \param   mesh is a const reference to the global data contained in the grid class
 * \param   solParam is a const reference to the user-set parameters contained in the parser class
 ********************************************************************************************************************************************
 */
multigrid_d2::multigrid_d2(const grid &mesh, const parser &solParam): poisson(mesh, solParam) {
    // CREATE THE MPI SUB-ARRAYS NECESSARY TO TRANSFER DATA ACROSS SUB-DOMAINS AT ALL MESH LEVELS
    createMGSubArrays();

    // INITIALIZE DIRICHLET BCs WHEN TESTING THE POISSON SOLVER
#ifdef TEST_POISSON
    initDirichlet();
#endif
}


void multigrid_d2::computeResidual() {
    tmpDataArray(vLevel) = 0.0;

    // Compute Laplacian of the pressure field and subtract it from the RHS of Poisson equation to obtain the residual
    // This residual is temporarily stored into tmpDataArray, from which it will be coarsened into residualData array.
#pragma omp parallel for num_threads(inputParams.nThreads) default(none)
    for (int i = 0; i <= xEnd(vLevel); ++i) {
        for (int k = 0; k <= zEnd(vLevel); ++k) {
            tmpDataArray(vLevel)(i, 0, k) =  residualData(vLevel)(i, 0, k) -
                         (xix2(vLevel)(i) * (pressureData(vLevel)(i + 1, 0, k) - 2.0*pressureData(vLevel)(i, 0, k) + pressureData(vLevel)(i - 1, 0, k))/(hx(vLevel)*hx(vLevel)) +
                          xixx(vLevel)(i) * (pressureData(vLevel)(i + 1, 0, k) - pressureData(vLevel)(i - 1, 0, k))/(2.0*hx(vLevel)) +
                          ztz2(vLevel)(k) * (pressureData(vLevel)(i, 0, k + 1) - 2.0*pressureData(vLevel)(i, 0, k) + pressureData(vLevel)(i, 0, k - 1))/(hz(vLevel)*hz(vLevel)) +
                          ztzz(vLevel)(k) * (pressureData(vLevel)(i, 0, k + 1) - pressureData(vLevel)(i, 0, k - 1))/(2.0*hz(vLevel)));
        }
    }

    updatePads(tmpDataArray);
}


void multigrid_d2::smooth(const int smoothCount) {
    tmpDataArray(vLevel) = 0.0;

    for(int n=0; n<smoothCount; ++n) {
        imposeBC();

        // WARNING: When using the gauss-seidel smoothing as written below, the edges of interior sub-domains after MPI decomposition will not have the updated values
        // As a result, the serial and parallel results will not match when using gauss-seidel smoothing
        if (inputParams.gsSmooth) {
            // GAUSS-SEIDEL ITERATIVE SMOOTHING
            for (int i = 0; i <= xEnd(vLevel); ++i) {
                for (int k = 0; k <= zEnd(vLevel); ++k) {
                    pressureData(vLevel)(i, 0, k) = (hz2(vLevel) * xix2(vLevel)(i) * (pressureData(vLevel)(i + 1, 0, k) + pressureData(vLevel)(i - 1, 0, k))*2.0 +
                                                     hz2(vLevel) * xixx(vLevel)(i) * (pressureData(vLevel)(i + 1, 0, k) - pressureData(vLevel)(i - 1, 0, k))*hx(vLevel) +
                                                     hx2(vLevel) * ztz2(vLevel)(k) * (pressureData(vLevel)(i, 0, k + 1) + pressureData(vLevel)(i, 0, k - 1))*2.0 +
                                                     hx2(vLevel) * ztzz(vLevel)(k) * (pressureData(vLevel)(i, 0, k + 1) - pressureData(vLevel)(i, 0, k - 1))*hz(vLevel) -
                                              2.0 * hzhx(vLevel) * residualData(vLevel)(i, 0, k))/
                                             (4.0 * (hz2(vLevel) * xix2(vLevel)(i) + hx2(vLevel)*ztz2(vLevel)(k)));
                }
            }
        } else {
            // JACOBI ITERATIVE SMOOTHING
#pragma omp parallel for num_threads(inputParams.nThreads) default(none)
            for (int i = 0; i <= xEnd(vLevel); ++i) {
                for (int k = 0; k <= zEnd(vLevel); ++k) {
                    tmpDataArray(vLevel)(i, 0, k) = (hz2(vLevel) * xix2(vLevel)(i) * (pressureData(vLevel)(i + 1, 0, k) + pressureData(vLevel)(i - 1, 0, k))*2.0 +
                                                     hz2(vLevel) * xixx(vLevel)(i) * (pressureData(vLevel)(i + 1, 0, k) - pressureData(vLevel)(i - 1, 0, k))*hx(vLevel) +
                                                     hx2(vLevel) * ztz2(vLevel)(k) * (pressureData(vLevel)(i, 0, k + 1) + pressureData(vLevel)(i, 0, k - 1))*2.0 +
                                                     hx2(vLevel) * ztzz(vLevel)(k) * (pressureData(vLevel)(i, 0, k + 1) - pressureData(vLevel)(i, 0, k - 1))*hz(vLevel) -
                                              2.0 * hzhx(vLevel) * residualData(vLevel)(i, 0, k))/
                                             (4.0 * (hz2(vLevel) * xix2(vLevel)(i) + hx2(vLevel)*ztz2(vLevel)(k)));
                }
            }

            swap(tmpDataArray, pressureData);
        }
    }

    imposeBC();
}


void multigrid_d2::solve() {
    int iterCount = 0;
    real tempValue, localMax, globalMax;

    while (true) {
        imposeBC();

        // GAUSS-SEIDEL ITERATIVE SOLVER
        for (int i = 0; i <= xEnd(vLevel); ++i) {
            for (int k = 0; k <= zEnd(vLevel); ++k) {
                pressureData(vLevel)(i, 0, k) = (hz2(vLevel) * xix2(vLevel)(i) * (pressureData(vLevel)(i + 1, 0, k) + pressureData(vLevel)(i - 1, 0, k))*2.0 +
                                                 hz2(vLevel) * xixx(vLevel)(i) * (pressureData(vLevel)(i + 1, 0, k) - pressureData(vLevel)(i - 1, 0, k))*hx(vLevel) +
                                                 hx2(vLevel) * ztz2(vLevel)(k) * (pressureData(vLevel)(i, 0, k + 1) + pressureData(vLevel)(i, 0, k - 1))*2.0 +
                                                 hx2(vLevel) * ztzz(vLevel)(k) * (pressureData(vLevel)(i, 0, k + 1) - pressureData(vLevel)(i, 0, k - 1))*hz(vLevel) -
                                          2.0 * hzhx(vLevel) * residualData(vLevel)(i, 0, k))/
                                         (4.0 * (hz2(vLevel) * xix2(vLevel)(i) + hx2(vLevel)*ztz2(vLevel)(k)));
            }
        }

        tempValue = 0.0;
        localMax = -1.0e-10;
        for (int i = 0; i <= xEnd(vLevel); ++i) {
            for (int k = 0; k <= zEnd(vLevel); ++k) {
                tempValue =  fabs(residualData(vLevel)(i, 0, k) -
                            (xix2(vLevel)(i) * (pressureData(vLevel)(i + 1, 0, k) - 2.0*pressureData(vLevel)(i, 0, k) + pressureData(vLevel)(i - 1, 0, k))/(hx(vLevel)*hx(vLevel)) +
                             xixx(vLevel)(i) * (pressureData(vLevel)(i + 1, 0, k) - pressureData(vLevel)(i - 1, 0, k))/(2.0*hx(vLevel)) +
                             ztz2(vLevel)(k) * (pressureData(vLevel)(i, 0, k + 1) - 2.0*pressureData(vLevel)(i, 0, k) + pressureData(vLevel)(i, 0, k - 1))/(hz(vLevel)*hz(vLevel)) +
                             ztzz(vLevel)(k) * (pressureData(vLevel)(i, 0, k + 1) - pressureData(vLevel)(i, 0, k - 1))/(2.0*hz(vLevel))));

                if (tempValue > localMax)
                    localMax = tempValue;
            }
        }

        MPI_Allreduce(&localMax, &globalMax, 1, MPI_FP_REAL, MPI_MAX, MPI_COMM_WORLD);

        if (globalMax < inputParams.mgTolerance) break;

        iterCount += 1;
        if (iterCount > maxCount) {
            if (inputParams.printResidual)
                if (mesh.rankData.rank == 0)
                    std::cout << "WARNING: Iterations for solution at coarsest level not converging." << std::endl;

            break;
        }
    }

    imposeBC();
}


void multigrid_d2::coarsen() {
    int pLevel;
    int i2, k2;

    pLevel = vLevel;
    vLevel += 1;

    for (int i = 0; i <= xEnd(vLevel); ++i) {
        i2 = i*2;
        for (int k = 0; k <= zEnd(vLevel); ++k) {
            k2 = k*2;
            residualData(vLevel)(i, 0, k) = (tmpDataArray(pLevel)(i2 + 1, 0, k2 + 1) + tmpDataArray(pLevel)(i2, 0, k2) +
                                             tmpDataArray(pLevel)(i2 + 1, 0, k2) + tmpDataArray(pLevel)(i2, 0, k2 + 1))/4;
        }
    }
}


void multigrid_d2::prolong() {
    int pLevel;
    int i2, k2;

    pLevel = vLevel;
    vLevel -= 1;

    pressureData(vLevel) = 0.0;

    for (int i = 0; i <= xEnd(vLevel); ++i) {
        i2 = i/2;
        for (int k = 0; k <= zEnd(vLevel); ++k) {
            k2 = k/2;
            pressureData(vLevel)(i, 0, k) = pressureData(pLevel)(i2, 0, k2);
        }
    }
}


real multigrid_d2::computeError(const int normOrder) {
    real residualVal = 0.0;
    real numValLoc = 0.0;
    real denValLoc = 0.0;
    real tempNum = 0.0;
    real tempDen = 0.0;

    // This function is called at the finest grid level only.

    // Problem with Koenig lookup is that when using the function abs with blitz arrays, it automatically computes
    // the absolute of the float values without hitch.
    // When replacing with computing absolute of individual array elements in a loop, ADL chooses a version of
    // abs in the STL which **rounds off** the number.
    // In this case, abs has to be replaced with fabs.
    for (int i = 0; i <= xEnd(0); ++i) {
        for (int k = 0; k <= zEnd(0); ++k) {
            tempNum = fabs((xix2(0)(i) * (pressureData(0)(i + 1, 0, k) - 2.0*pressureData(0)(i, 0, k) + pressureData(0)(i - 1, 0, k))/hx2(0) +
                            xixx(0)(i) * (pressureData(0)(i + 1, 0, k) - pressureData(0)(i - 1, 0, k))/(2.0*hx(0)) +
                            ztz2(0)(k) * (pressureData(0)(i, 0, k + 1) - 2.0*pressureData(0)(i, 0, k) + pressureData(0)(i, 0, k - 1))/hz2(0) +
                            ztzz(0)(k) * (pressureData(0)(i, 0, k + 1) - pressureData(0)(i, 0, k - 1))/(2.0*hz(0))) - residualData(0)(i, 0, k));

            tempDen = fabs(residualData(0)(i, 0, k));

            switch (normOrder) {
                case 0: // L-Infinity Norm
                    if (tempNum > numValLoc) numValLoc = tempNum;
                    if (tempDen > denValLoc) denValLoc = tempDen;
                    break;
                case 1: // L-1 Norm
                    numValLoc += tempNum;
                    denValLoc += tempDen;
                    break;
                case 2: // L-2 Norm
                    numValLoc += tempNum*tempNum;
                    denValLoc += tempDen*tempDen;
                    break;
            }
        }
    }

    real numValGlo = 0.0;
    real denValGlo = 0.0;
    int pointCount = mesh.totalPoints;
    switch (normOrder) {
        case 0:     // L-Infinity Norm
            MPI_Allreduce(&numValLoc, &numValGlo, 1, MPI_FP_REAL, MPI_MAX, MPI_COMM_WORLD);
            MPI_Allreduce(&denValLoc, &denValGlo, 1, MPI_FP_REAL, MPI_MAX, MPI_COMM_WORLD);

            if (denValGlo) {
                residualVal = numValGlo/denValGlo;
            } else {
                residualVal = numValGlo;
            }
            break;
        case 1:     // L-1 Norm
            MPI_Allreduce(&numValLoc, &numValGlo, 1, MPI_FP_REAL, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&denValLoc, &denValGlo, 1, MPI_FP_REAL, MPI_SUM, MPI_COMM_WORLD);

            if (denValGlo) {
                residualVal = numValGlo/denValGlo;
            } else {
                residualVal = numValGlo/pointCount;
            }
            break;
        case 2:     // L-2 Norm
            MPI_Allreduce(&numValLoc, &numValGlo, 1, MPI_FP_REAL, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&denValLoc, &denValGlo, 1, MPI_FP_REAL, MPI_SUM, MPI_COMM_WORLD);

            if (denValGlo) {
                residualVal = sqrt(numValGlo/pointCount)/sqrt(denValGlo/pointCount);
            } else {
                residualVal = sqrt(numValGlo/pointCount);
            }
            break;
    }

    return residualVal;
}


void multigrid_d2::createMGSubArrays() {
    int count;

    recvStatus.resize(2);
    recvRequest.resize(2);

    xMGArray.resize(inputParams.vcDepth + 1);
    mgSendLft.resize(inputParams.vcDepth + 1);        mgSendRgt.resize(inputParams.vcDepth + 1);
    mgRecvLft.resize(inputParams.vcDepth + 1);        mgRecvRgt.resize(inputParams.vcDepth + 1);

    for(int n=0; n<=inputParams.vcDepth; ++n) {
        // CREATE X_MG_ARRAY DATATYPE
        count = stagFull(n).ubound(2) + 2;

        MPI_Type_contiguous(count, MPI_FP_REAL, &xMGArray(n));
        MPI_Type_commit(&xMGArray(n));

        // SET STARTING INDICES OF MEMORY LOCATIONS FROM WHERE TO READ (SEND) AND WRITE (RECEIVE) DATA
        mgSendLft(n) =  0, 0, -1;
        mgRecvLft(n) = -1, 0, -1;
        mgSendRgt(n) = stagCore(n).ubound(0), 0, -1;
        mgRecvRgt(n) = stagCore(n).ubound(0) + 1, 0, -1;
    }
}


void multigrid_d2::initDirichlet() {
    real xDist, zDist;

    // Generate the walls as 1D Blitz arrays
    xWall.resize(stagFull(0).ubound(2) - stagFull(0).lbound(2) + 1);
    xWall.reindexSelf(stagFull(0).lbound(2));
    xWall = 0.0;

    zWall.resize(stagFull(0).ubound(0) - stagFull(0).lbound(0) + 1);
    zWall.reindexSelf(stagFull(0).lbound(0));
    zWall = 0.0;

    // Compute values at the walls using the (r^2)/4 formula
    // Along X-direction - Left and Right Walls
    xDist = mesh.inputParams.Lx/2.0;
    for (int k=0; k<=stagCore(0).ubound(2); ++k) {
        zDist = mesh.z(k) - 0.5;

        xWall(k) = (xDist*xDist + zDist*zDist)/4.0;
    }

    // Along Z-direction - Top and Bottom Walls
    zDist = mesh.inputParams.Lz/2.0;
    for (int i=0; i<=stagCore(0).ubound(0); ++i) {
        xDist = mesh.x(i) - 0.5;

        zWall(i) = (xDist*xDist + zDist*zDist)/4.0;
    }
}


void multigrid_d2::imposeBC() {
    // THIS FLAG WILL SWITCH BETWEEN TESTING NEUMANN BC AND DIRICHLET BC FOR TEST_POISSON RUNS.
    // SINCE THIS IS NOT A SOLVER-WIDE FEATURE (ONLY WORKS FOR UNIFORM-GRID 2D CASE), IT IS
    // CURRENTLY OFFERED ONLY AS A HIDDEN FLAG WITHIN THE CODE THAT WILL NOT BE FOUND UNLESS
    // YOU GO LOOKING FOR TROUBLE.
#ifdef TEST_POISSON
    bool testNeumann = false;
#endif

    // FOR PARALLEL RUNS, FIRST UPDATE GHOST POINTS OF MPI SUB-DOMAINS
    updatePads(pressureData);

    if (not inputParams.xPer) {
#ifdef TEST_POISSON
        // DIRICHLET/NEUMANN BOUNDARY CONDITION AT LEFT AND RIGHT WALLS
        if (zeroBC) {
            if (mesh.rankData.xRank == 0) {
                if (testNeumann) {
                    pressureData(vLevel)(-1, 0, all) = pressureData(vLevel)(0, 0, all);
                } else {
                    pressureData(vLevel)(-1, 0, all) = -pressureData(vLevel)(0, 0, all);
                }
            }

            if (mesh.rankData.xRank == mesh.rankData.npX - 1) {
                if (testNeumann) {
                    pressureData(vLevel)(stagCore(vLevel).ubound(0) + 1, 0, all) = pressureData(vLevel)(stagCore(vLevel).ubound(0), 0, all);
                } else {
                    pressureData(vLevel)(stagCore(vLevel).ubound(0) + 1, 0, all) = -pressureData(vLevel)(stagCore(vLevel).ubound(0), 0, all);
                }
            }
        } else {
            if (mesh.rankData.xRank == 0) {
                if (testNeumann) {
                    pressureData(vLevel)(-1, 0, all) = 0.5*hx(vLevel) + pressureData(vLevel)(0, 0, all);
                } else {
                    pressureData(vLevel)(-1, 0, all) = 2.0*xWall(all) - pressureData(vLevel)(0, 0, all);
                }
            }

            if (mesh.rankData.xRank == mesh.rankData.npX - 1) {
                if (testNeumann) {
                    pressureData(vLevel)(stagCore(vLevel).ubound(0) + 1, 0, all) = 0.5*hx(vLevel) + pressureData(vLevel)(stagCore(vLevel).ubound(0), 0, all);
                } else {
                    pressureData(vLevel)(stagCore(vLevel).ubound(0) + 1, 0, all) = 2.0*xWall(all) - pressureData(vLevel)(stagCore(vLevel).ubound(0), 0, all);
                }
            }
        }
#else
        // NEUMANN BOUNDARY CONDITION AT LEFT AND RIGHT WALLS
        if (mesh.rankData.xRank == 0) {
            pressureData(vLevel)(-1, 0, all) = pressureData(vLevel)(0, 0, all);
        }

        if (mesh.rankData.xRank == mesh.rankData.npX - 1) {
            pressureData(vLevel)(stagCore(vLevel).ubound(0) + 1, 0, all) = pressureData(vLevel)(stagCore(vLevel).ubound(0), 0, all);
        }
#endif
    } // PERIODIC BOUNDARY CONDITIONS ARE AUTOMATICALLY IMPOSED BY PERIODIC DATA TRANSFER ACROSS PROCESSORS THROUGH updatePads()

    if (inputParams.zPer) {
        // PERIODIC BOUNDARY CONDITION AT BOTTOM WALL
        pressureData(vLevel)(all, 0, -1) = pressureData(vLevel)(all, 0, stagCore(vLevel).ubound(2));

        // PERIODIC BOUNDARY CONDITION AT TOP WALL
        pressureData(vLevel)(all, 0, stagCore(vLevel).ubound(2) + 1) = pressureData(vLevel)(all, 0, 0);

    } else {
#ifdef TEST_POISSON
        // DIRICHLET/NEUMANN BOUNDARY CONDITION AT BOTTOM AND TOP WALLS
        if (zeroBC) {
            if (testNeumann) {
                pressureData(vLevel)(all, 0, -1) = pressureData(vLevel)(all, 0, 0);
            } else {
                pressureData(vLevel)(all, 0, -1) = -pressureData(vLevel)(all, 0, 0);
            }

            // WHETHER testNeumann IS ENABLED OR NOT, THE TOP WALL HAS DIRICHLET BC SINCE ALL 4 WALLS SHOULD NOT HAVE NEUMANN BC
            pressureData(vLevel)(all, 0, stagCore(vLevel).ubound(2) + 1) = -pressureData(vLevel)(all, 0, stagCore(vLevel).ubound(2));
        } else {
            if (testNeumann) {
                pressureData(vLevel)(all, 0, -1) = 0.5*hz(vLevel) + pressureData(vLevel)(all, 0, 0);
            } else {
                pressureData(vLevel)(all, 0, -1) = 2.0*zWall(all) - pressureData(vLevel)(all, 0, 0);
            }

            // WHETHER testNeumann IS ENABLED OR NOT, THE TOP WALL HAS DIRICHLET BC SINCE ALL 4 WALLS SHOULD NOT HAVE NEUMANN BC
            pressureData(vLevel)(all, 0, stagCore(vLevel).ubound(2) + 1) = 2.0*zWall(all) - pressureData(vLevel)(all, 0, stagCore(vLevel).ubound(2));
        }
#else
        // NEUMANN BOUNDARY CONDITION AT BOTTOM AND TOP WALLS
        pressureData(vLevel)(all, 0, -1) = pressureData(vLevel)(all, 0, 0);

        pressureData(vLevel)(all, 0, stagCore(vLevel).ubound(2) + 1) = pressureData(vLevel)(all, 0, stagCore(vLevel).ubound(2));
#endif
    }
}


void multigrid_d2::updatePads(blitz::Array<blitz::Array<real, 3>, 1> &data) {
    recvRequest = MPI_REQUEST_NULL;

    // TRANSFER DATA FROM NEIGHBOURING CELL TO IMPOSE SUB-DOMAIN BOUNDARY CONDITIONS
    MPI_Irecv(&(data(vLevel)(mgRecvLft(vLevel))), 1, xMGArray(vLevel), mesh.rankData.faceRanks(0), 1, MPI_COMM_WORLD, &recvRequest(0));
    MPI_Irecv(&(data(vLevel)(mgRecvRgt(vLevel))), 1, xMGArray(vLevel), mesh.rankData.faceRanks(1), 2, MPI_COMM_WORLD, &recvRequest(1));

    MPI_Send(&(data(vLevel)(mgSendLft(vLevel))), 1, xMGArray(vLevel), mesh.rankData.faceRanks(0), 2, MPI_COMM_WORLD);
    MPI_Send(&(data(vLevel)(mgSendRgt(vLevel))), 1, xMGArray(vLevel), mesh.rankData.faceRanks(1), 1, MPI_COMM_WORLD);

    MPI_Waitall(2, recvRequest.dataFirst(), recvStatus.dataFirst());
}

