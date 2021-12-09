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
/*! \file poisson3.cc
 *
 *  \brief Definitions for functions of class poisson for 3D
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
 * \brief   Constructor of the multigrid_d3 class derived from the poisson class
 *
 *          The constructor of the derived multigrid_d3 class frst calls the base poisson class with the arguments passed to it.
 *          Most of the array initializations are performed in the constructor of the base class.
 *          Initializations unique to 3D implementation of multi-grid solver are then done here.
 *          Primarily this involves generating the MPI datatypes for data transfer between sub-domain boundaries.
 *          When testing the Poisson solver with Dirichlet BC, the analytical 3D solution for the test is also calculated here
 *          through a call to initDirichlet() function.
 *
 * \param   mesh is a const reference to the global data contained in the grid class
 * \param   solParam is a const reference to the user-set parameters contained in the parser class
 ********************************************************************************************************************************************
 */
multigrid_d3::multigrid_d3(const grid &mesh, const parser &solParam): poisson(mesh, solParam) {
    // CREATE THE MPI SUB-ARRAYS NECESSARY TO TRANSFER DATA ACROSS SUB-DOMAINS AT ALL MESH LEVELS
    createMGSubArrays();

    // INITIALIZE DIRICHLET BCs WHEN TESTING THE POISSON SOLVER
#ifdef TEST_POISSON
    initDirichlet();
#endif
}


void multigrid_d3::computeResidual() {
    tmp(vLevel) = 0.0;

    // Compute Laplacian of the pressure field and subtract it from the RHS of Poisson equation to obtain the residual
    // This residual is temporarily stored into tmp, from which it will be coarsened into rhs array.
    // Needed update: Substitute the below OpenMP parallel loop with vectorized Blitz operation and check for speed increase.
#pragma omp parallel for num_threads(inputParams.nThreads) default(none)
    for (int i = 0; i <= xEnd(vLevel); ++i) {
        for (int j = 0; j <= yEnd(vLevel); ++j) {
            for (int k = 0; k <= zEnd(vLevel); ++k) {
                tmp(vLevel)(i, j, k) =  rhs(vLevel)(i, j, k) -
                             (xix2(vLevel)(i) * ihx2(vLevel) * (lhs(vLevel)(i + 1, j, k) - 2.0*lhs(vLevel)(i, j, k) + lhs(vLevel)(i - 1, j, k)) +
                              xixx(vLevel)(i) * i2hx(vLevel) * (lhs(vLevel)(i + 1, j, k) - lhs(vLevel)(i - 1, j, k)) +
                              ety2(vLevel)(j) * ihy2(vLevel) * (lhs(vLevel)(i, j + 1, k) - 2.0*lhs(vLevel)(i, j, k) + lhs(vLevel)(i, j - 1, k)) +
                              etyy(vLevel)(j) * i2hy(vLevel) * (lhs(vLevel)(i, j + 1, k) - lhs(vLevel)(i, j - 1, k)) +
                              ztz2(vLevel)(k) * ihz2(vLevel) * (lhs(vLevel)(i, j, k + 1) - 2.0*lhs(vLevel)(i, j, k) + lhs(vLevel)(i, j, k - 1)) +
                              ztzz(vLevel)(k) * i2hz(vLevel) * (lhs(vLevel)(i, j, k + 1) - lhs(vLevel)(i, j, k - 1)));
            }
        }
    }

    updateFull(tmp);
}


void multigrid_d3::smooth(const int smoothCount) {
    tmp(vLevel) = 0.0;

    for(int n=0; n<smoothCount; ++n) {
        imposeBC();

        // RED-BLACK GAUSS-SEIDEL
        // UPDATE RED CELLS
        // 0, 0, 0 CONFIGURATION
#pragma omp parallel for num_threads(inputParams.nThreads) default(none)
        for (int i = 0; i <= xEnd(vLevel); i+=2) {
            for (int j = 0; j <= yEnd(vLevel); j+=2) {
                for (int k = 0; k <= zEnd(vLevel); k+=2) {
                    lhs(vLevel)(i, j, k) = (1.0 - sorParam) * lhs(vLevel)(i, j, k) +
                                           (xix2(vLevel)(i) * ihx2(vLevel) * (lhs(vLevel)(i + 1, j, k) + lhs(vLevel)(i - 1, j, k)) +
                                            xixx(vLevel)(i) * i2hx(vLevel) * (lhs(vLevel)(i + 1, j, k) - lhs(vLevel)(i - 1, j, k)) +
                                            ety2(vLevel)(j) * ihy2(vLevel) * (lhs(vLevel)(i, j + 1, k) + lhs(vLevel)(i, j - 1, k)) +
                                            etyy(vLevel)(j) * i2hy(vLevel) * (lhs(vLevel)(i, j + 1, k) - lhs(vLevel)(i, j - 1, k)) +
                                            ztz2(vLevel)(k) * ihz2(vLevel) * (lhs(vLevel)(i, j, k + 1) + lhs(vLevel)(i, j, k - 1)) +
                                            ztzz(vLevel)(k) * i2hz(vLevel) * (lhs(vLevel)(i, j, k + 1) - lhs(vLevel)(i, j, k - 1)) -
                                                rhs(vLevel)(i, j, k)) * sorParam / (2.0 * (ihx2(vLevel)*xix2(vLevel)(i) + ihy2(vLevel)*ety2(vLevel)(j) + ihz2(vLevel)*ztz2(vLevel)(k)));
                }
            }
        }

        // 1, 1, 0 CONFIGURATION
#pragma omp parallel for num_threads(inputParams.nThreads) default(none)
        for (int i = 1; i <= xEnd(vLevel); i+=2) {
            for (int j = 1; j <= yEnd(vLevel); j+=2) {
                for (int k = 0; k <= zEnd(vLevel); k+=2) {
                    lhs(vLevel)(i, j, k) = (1.0 - sorParam) * lhs(vLevel)(i, j, k) +
                                           (xix2(vLevel)(i) * ihx2(vLevel) * (lhs(vLevel)(i + 1, j, k) + lhs(vLevel)(i - 1, j, k)) +
                                            xixx(vLevel)(i) * i2hx(vLevel) * (lhs(vLevel)(i + 1, j, k) - lhs(vLevel)(i - 1, j, k)) +
                                            ety2(vLevel)(j) * ihy2(vLevel) * (lhs(vLevel)(i, j + 1, k) + lhs(vLevel)(i, j - 1, k)) +
                                            etyy(vLevel)(j) * i2hy(vLevel) * (lhs(vLevel)(i, j + 1, k) - lhs(vLevel)(i, j - 1, k)) +
                                            ztz2(vLevel)(k) * ihz2(vLevel) * (lhs(vLevel)(i, j, k + 1) + lhs(vLevel)(i, j, k - 1)) +
                                            ztzz(vLevel)(k) * i2hz(vLevel) * (lhs(vLevel)(i, j, k + 1) - lhs(vLevel)(i, j, k - 1)) -
                                                rhs(vLevel)(i, j, k)) * sorParam / (2.0 * (ihx2(vLevel)*xix2(vLevel)(i) + ihy2(vLevel)*ety2(vLevel)(j) + ihz2(vLevel)*ztz2(vLevel)(k)));
                }
            }
        }

        // 1, 0, 1 CONFIGURATION
#pragma omp parallel for num_threads(inputParams.nThreads) default(none)
        for (int i = 1; i <= xEnd(vLevel); i+=2) {
            for (int j = 0; j <= yEnd(vLevel); j+=2) {
                for (int k = 1; k <= zEnd(vLevel); k+=2) {
                    lhs(vLevel)(i, j, k) = (1.0 - sorParam) * lhs(vLevel)(i, j, k) +
                                           (xix2(vLevel)(i) * ihx2(vLevel) * (lhs(vLevel)(i + 1, j, k) + lhs(vLevel)(i - 1, j, k)) +
                                            xixx(vLevel)(i) * i2hx(vLevel) * (lhs(vLevel)(i + 1, j, k) - lhs(vLevel)(i - 1, j, k)) +
                                            ety2(vLevel)(j) * ihy2(vLevel) * (lhs(vLevel)(i, j + 1, k) + lhs(vLevel)(i, j - 1, k)) +
                                            etyy(vLevel)(j) * i2hy(vLevel) * (lhs(vLevel)(i, j + 1, k) - lhs(vLevel)(i, j - 1, k)) +
                                            ztz2(vLevel)(k) * ihz2(vLevel) * (lhs(vLevel)(i, j, k + 1) + lhs(vLevel)(i, j, k - 1)) +
                                            ztzz(vLevel)(k) * i2hz(vLevel) * (lhs(vLevel)(i, j, k + 1) - lhs(vLevel)(i, j, k - 1)) -
                                                rhs(vLevel)(i, j, k)) * sorParam / (2.0 * (ihx2(vLevel)*xix2(vLevel)(i) + ihy2(vLevel)*ety2(vLevel)(j) + ihz2(vLevel)*ztz2(vLevel)(k)));
                }
            }
        }

        // 0, 1, 1 CONFIGURATION
#pragma omp parallel for num_threads(inputParams.nThreads) default(none)
        for (int i = 0; i <= xEnd(vLevel); i+=2) {
            for (int j = 1; j <= yEnd(vLevel); j+=2) {
                for (int k = 1; k <= zEnd(vLevel); k+=2) {
                    lhs(vLevel)(i, j, k) = (1.0 - sorParam) * lhs(vLevel)(i, j, k) +
                                           (xix2(vLevel)(i) * ihx2(vLevel) * (lhs(vLevel)(i + 1, j, k) + lhs(vLevel)(i - 1, j, k)) +
                                            xixx(vLevel)(i) * i2hx(vLevel) * (lhs(vLevel)(i + 1, j, k) - lhs(vLevel)(i - 1, j, k)) +
                                            ety2(vLevel)(j) * ihy2(vLevel) * (lhs(vLevel)(i, j + 1, k) + lhs(vLevel)(i, j - 1, k)) +
                                            etyy(vLevel)(j) * i2hy(vLevel) * (lhs(vLevel)(i, j + 1, k) - lhs(vLevel)(i, j - 1, k)) +
                                            ztz2(vLevel)(k) * ihz2(vLevel) * (lhs(vLevel)(i, j, k + 1) + lhs(vLevel)(i, j, k - 1)) +
                                            ztzz(vLevel)(k) * i2hz(vLevel) * (lhs(vLevel)(i, j, k + 1) - lhs(vLevel)(i, j, k - 1)) -
                                                rhs(vLevel)(i, j, k)) * sorParam / (2.0 * (ihx2(vLevel)*xix2(vLevel)(i) + ihy2(vLevel)*ety2(vLevel)(j) + ihz2(vLevel)*ztz2(vLevel)(k)));
                }
            }
        }

        // UPDATE OF RED CELLS COMPLETE. UPDATE SUB-DOMAIN FACES NOW
        updateFace(lhs);

        // UPDATE BLACK CELLS
        // 1, 0, 0 CONFIGURATION
#pragma omp parallel for num_threads(inputParams.nThreads) default(none)
        for (int i = 1; i <= xEnd(vLevel); i+=2) {
            for (int j = 0; j <= yEnd(vLevel); j+=2) {
                for (int k = 0; k <= zEnd(vLevel); k+=2) {
                    lhs(vLevel)(i, j, k) = (1.0 - sorParam) * lhs(vLevel)(i, j, k) +
                                           (xix2(vLevel)(i) * ihx2(vLevel) * (lhs(vLevel)(i + 1, j, k) + lhs(vLevel)(i - 1, j, k)) +
                                            xixx(vLevel)(i) * i2hx(vLevel) * (lhs(vLevel)(i + 1, j, k) - lhs(vLevel)(i - 1, j, k)) +
                                            ety2(vLevel)(j) * ihy2(vLevel) * (lhs(vLevel)(i, j + 1, k) + lhs(vLevel)(i, j - 1, k)) +
                                            etyy(vLevel)(j) * i2hy(vLevel) * (lhs(vLevel)(i, j + 1, k) - lhs(vLevel)(i, j - 1, k)) +
                                            ztz2(vLevel)(k) * ihz2(vLevel) * (lhs(vLevel)(i, j, k + 1) + lhs(vLevel)(i, j, k - 1)) +
                                            ztzz(vLevel)(k) * i2hz(vLevel) * (lhs(vLevel)(i, j, k + 1) - lhs(vLevel)(i, j, k - 1)) -
                                                rhs(vLevel)(i, j, k)) * sorParam / (2.0 * (ihx2(vLevel)*xix2(vLevel)(i) + ihy2(vLevel)*ety2(vLevel)(j) + ihz2(vLevel)*ztz2(vLevel)(k)));
                }
            }
        }

        // 0, 1, 0 CONFIGURATION
#pragma omp parallel for num_threads(inputParams.nThreads) default(none)
        for (int i = 0; i <= xEnd(vLevel); i+=2) {
            for (int j = 1; j <= yEnd(vLevel); j+=2) {
                for (int k = 0; k <= zEnd(vLevel); k+=2) {
                    lhs(vLevel)(i, j, k) = (1.0 - sorParam) * lhs(vLevel)(i, j, k) +
                                           (xix2(vLevel)(i) * ihx2(vLevel) * (lhs(vLevel)(i + 1, j, k) + lhs(vLevel)(i - 1, j, k)) +
                                            xixx(vLevel)(i) * i2hx(vLevel) * (lhs(vLevel)(i + 1, j, k) - lhs(vLevel)(i - 1, j, k)) +
                                            ety2(vLevel)(j) * ihy2(vLevel) * (lhs(vLevel)(i, j + 1, k) + lhs(vLevel)(i, j - 1, k)) +
                                            etyy(vLevel)(j) * i2hy(vLevel) * (lhs(vLevel)(i, j + 1, k) - lhs(vLevel)(i, j - 1, k)) +
                                            ztz2(vLevel)(k) * ihz2(vLevel) * (lhs(vLevel)(i, j, k + 1) + lhs(vLevel)(i, j, k - 1)) +
                                            ztzz(vLevel)(k) * i2hz(vLevel) * (lhs(vLevel)(i, j, k + 1) - lhs(vLevel)(i, j, k - 1)) -
                                                rhs(vLevel)(i, j, k)) * sorParam / (2.0 * (ihx2(vLevel)*xix2(vLevel)(i) + ihy2(vLevel)*ety2(vLevel)(j) + ihz2(vLevel)*ztz2(vLevel)(k)));
                }
            }
        }

        // 0, 0, 1 CONFIGURATION
#pragma omp parallel for num_threads(inputParams.nThreads) default(none)
        for (int i = 0; i <= xEnd(vLevel); i+=2) {
            for (int j = 0; j <= yEnd(vLevel); j+=2) {
                for (int k = 1; k <= zEnd(vLevel); k+=2) {
                    lhs(vLevel)(i, j, k) = (1.0 - sorParam) * lhs(vLevel)(i, j, k) +
                                           (xix2(vLevel)(i) * ihx2(vLevel) * (lhs(vLevel)(i + 1, j, k) + lhs(vLevel)(i - 1, j, k)) +
                                            xixx(vLevel)(i) * i2hx(vLevel) * (lhs(vLevel)(i + 1, j, k) - lhs(vLevel)(i - 1, j, k)) +
                                            ety2(vLevel)(j) * ihy2(vLevel) * (lhs(vLevel)(i, j + 1, k) + lhs(vLevel)(i, j - 1, k)) +
                                            etyy(vLevel)(j) * i2hy(vLevel) * (lhs(vLevel)(i, j + 1, k) - lhs(vLevel)(i, j - 1, k)) +
                                            ztz2(vLevel)(k) * ihz2(vLevel) * (lhs(vLevel)(i, j, k + 1) + lhs(vLevel)(i, j, k - 1)) +
                                            ztzz(vLevel)(k) * i2hz(vLevel) * (lhs(vLevel)(i, j, k + 1) - lhs(vLevel)(i, j, k - 1)) -
                                                rhs(vLevel)(i, j, k)) * sorParam / (2.0 * (ihx2(vLevel)*xix2(vLevel)(i) + ihy2(vLevel)*ety2(vLevel)(j) + ihz2(vLevel)*ztz2(vLevel)(k)));
                }
            }
        }

        // 1, 1, 1 CONFIGURATION
#pragma omp parallel for num_threads(inputParams.nThreads) default(none)
        for (int i = 1; i <= xEnd(vLevel); i+=2) {
            for (int j = 1; j <= yEnd(vLevel); j+=2) {
                for (int k = 1; k <= zEnd(vLevel); k+=2) {
                    lhs(vLevel)(i, j, k) = (1.0 - sorParam) * lhs(vLevel)(i, j, k) +
                                           (xix2(vLevel)(i) * ihx2(vLevel) * (lhs(vLevel)(i + 1, j, k) + lhs(vLevel)(i - 1, j, k)) +
                                            xixx(vLevel)(i) * i2hx(vLevel) * (lhs(vLevel)(i + 1, j, k) - lhs(vLevel)(i - 1, j, k)) +
                                            ety2(vLevel)(j) * ihy2(vLevel) * (lhs(vLevel)(i, j + 1, k) + lhs(vLevel)(i, j - 1, k)) +
                                            etyy(vLevel)(j) * i2hy(vLevel) * (lhs(vLevel)(i, j + 1, k) - lhs(vLevel)(i, j - 1, k)) +
                                            ztz2(vLevel)(k) * ihz2(vLevel) * (lhs(vLevel)(i, j, k + 1) + lhs(vLevel)(i, j, k - 1)) +
                                            ztzz(vLevel)(k) * i2hz(vLevel) * (lhs(vLevel)(i, j, k + 1) - lhs(vLevel)(i, j, k - 1)) -
                                                rhs(vLevel)(i, j, k)) * sorParam / (2.0 * (ihx2(vLevel)*xix2(vLevel)(i) + ihy2(vLevel)*ety2(vLevel)(j) + ihz2(vLevel)*ztz2(vLevel)(k)));
                }
            }
        }
    }

    imposeBC();
}


void multigrid_d3::solve() {
    int iterCount = 0;
    real tempValue, localMax, globalMax;

    while (true) {
        imposeBC();

        // RED-BLACK GAUSS-SEIDEL
        // UPDATE RED CELLS
        // 0, 0, 0 CONFIGURATION
#pragma omp parallel for num_threads(inputParams.nThreads) default(none)
        for (int i = 0; i <= xEnd(vLevel); i+=2) {
            for (int j = 0; j <= yEnd(vLevel); j+=2) {
                for (int k = 0; k <= zEnd(vLevel); k+=2) {
                    lhs(vLevel)(i, j, k) = (1.0 - sorParam) * lhs(vLevel)(i, j, k) +
                                           (xix2(vLevel)(i) * ihx2(vLevel) * (lhs(vLevel)(i + 1, j, k) + lhs(vLevel)(i - 1, j, k)) +
                                            xixx(vLevel)(i) * i2hx(vLevel) * (lhs(vLevel)(i + 1, j, k) - lhs(vLevel)(i - 1, j, k)) +
                                            ety2(vLevel)(j) * ihy2(vLevel) * (lhs(vLevel)(i, j + 1, k) + lhs(vLevel)(i, j - 1, k)) +
                                            etyy(vLevel)(j) * i2hy(vLevel) * (lhs(vLevel)(i, j + 1, k) - lhs(vLevel)(i, j - 1, k)) +
                                            ztz2(vLevel)(k) * ihz2(vLevel) * (lhs(vLevel)(i, j, k + 1) + lhs(vLevel)(i, j, k - 1)) +
                                            ztzz(vLevel)(k) * i2hz(vLevel) * (lhs(vLevel)(i, j, k + 1) - lhs(vLevel)(i, j, k - 1)) -
                                                rhs(vLevel)(i, j, k)) * sorParam / (2.0 * (ihx2(vLevel)*xix2(vLevel)(i) + ihy2(vLevel)*ety2(vLevel)(j) + ihz2(vLevel)*ztz2(vLevel)(k)));
                }
            }
        }

        // 1, 1, 0 CONFIGURATION
#pragma omp parallel for num_threads(inputParams.nThreads) default(none)
        for (int i = 1; i <= xEnd(vLevel); i+=2) {
            for (int j = 1; j <= yEnd(vLevel); j+=2) {
                for (int k = 0; k <= zEnd(vLevel); k+=2) {
                    lhs(vLevel)(i, j, k) = (1.0 - sorParam) * lhs(vLevel)(i, j, k) +
                                           (xix2(vLevel)(i) * ihx2(vLevel) * (lhs(vLevel)(i + 1, j, k) + lhs(vLevel)(i - 1, j, k)) +
                                            xixx(vLevel)(i) * i2hx(vLevel) * (lhs(vLevel)(i + 1, j, k) - lhs(vLevel)(i - 1, j, k)) +
                                            ety2(vLevel)(j) * ihy2(vLevel) * (lhs(vLevel)(i, j + 1, k) + lhs(vLevel)(i, j - 1, k)) +
                                            etyy(vLevel)(j) * i2hy(vLevel) * (lhs(vLevel)(i, j + 1, k) - lhs(vLevel)(i, j - 1, k)) +
                                            ztz2(vLevel)(k) * ihz2(vLevel) * (lhs(vLevel)(i, j, k + 1) + lhs(vLevel)(i, j, k - 1)) +
                                            ztzz(vLevel)(k) * i2hz(vLevel) * (lhs(vLevel)(i, j, k + 1) - lhs(vLevel)(i, j, k - 1)) -
                                                rhs(vLevel)(i, j, k)) * sorParam / (2.0 * (ihx2(vLevel)*xix2(vLevel)(i) + ihy2(vLevel)*ety2(vLevel)(j) + ihz2(vLevel)*ztz2(vLevel)(k)));
                }
            }
        }

        // 1, 0, 1 CONFIGURATION
#pragma omp parallel for num_threads(inputParams.nThreads) default(none)
        for (int i = 1; i <= xEnd(vLevel); i+=2) {
            for (int j = 0; j <= yEnd(vLevel); j+=2) {
                for (int k = 1; k <= zEnd(vLevel); k+=2) {
                    lhs(vLevel)(i, j, k) = (1.0 - sorParam) * lhs(vLevel)(i, j, k) +
                                           (xix2(vLevel)(i) * ihx2(vLevel) * (lhs(vLevel)(i + 1, j, k) + lhs(vLevel)(i - 1, j, k)) +
                                            xixx(vLevel)(i) * i2hx(vLevel) * (lhs(vLevel)(i + 1, j, k) - lhs(vLevel)(i - 1, j, k)) +
                                            ety2(vLevel)(j) * ihy2(vLevel) * (lhs(vLevel)(i, j + 1, k) + lhs(vLevel)(i, j - 1, k)) +
                                            etyy(vLevel)(j) * i2hy(vLevel) * (lhs(vLevel)(i, j + 1, k) - lhs(vLevel)(i, j - 1, k)) +
                                            ztz2(vLevel)(k) * ihz2(vLevel) * (lhs(vLevel)(i, j, k + 1) + lhs(vLevel)(i, j, k - 1)) +
                                            ztzz(vLevel)(k) * i2hz(vLevel) * (lhs(vLevel)(i, j, k + 1) - lhs(vLevel)(i, j, k - 1)) -
                                                rhs(vLevel)(i, j, k)) * sorParam / (2.0 * (ihx2(vLevel)*xix2(vLevel)(i) + ihy2(vLevel)*ety2(vLevel)(j) + ihz2(vLevel)*ztz2(vLevel)(k)));
                }
            }
        }

        // 0, 1, 1 CONFIGURATION
#pragma omp parallel for num_threads(inputParams.nThreads) default(none)
        for (int i = 0; i <= xEnd(vLevel); i+=2) {
            for (int j = 1; j <= yEnd(vLevel); j+=2) {
                for (int k = 1; k <= zEnd(vLevel); k+=2) {
                    lhs(vLevel)(i, j, k) = (1.0 - sorParam) * lhs(vLevel)(i, j, k) +
                                           (xix2(vLevel)(i) * ihx2(vLevel) * (lhs(vLevel)(i + 1, j, k) + lhs(vLevel)(i - 1, j, k)) +
                                            xixx(vLevel)(i) * i2hx(vLevel) * (lhs(vLevel)(i + 1, j, k) - lhs(vLevel)(i - 1, j, k)) +
                                            ety2(vLevel)(j) * ihy2(vLevel) * (lhs(vLevel)(i, j + 1, k) + lhs(vLevel)(i, j - 1, k)) +
                                            etyy(vLevel)(j) * i2hy(vLevel) * (lhs(vLevel)(i, j + 1, k) - lhs(vLevel)(i, j - 1, k)) +
                                            ztz2(vLevel)(k) * ihz2(vLevel) * (lhs(vLevel)(i, j, k + 1) + lhs(vLevel)(i, j, k - 1)) +
                                            ztzz(vLevel)(k) * i2hz(vLevel) * (lhs(vLevel)(i, j, k + 1) - lhs(vLevel)(i, j, k - 1)) -
                                                rhs(vLevel)(i, j, k)) * sorParam / (2.0 * (ihx2(vLevel)*xix2(vLevel)(i) + ihy2(vLevel)*ety2(vLevel)(j) + ihz2(vLevel)*ztz2(vLevel)(k)));
                }
            }
        }

        // UPDATE OF RED CELLS COMPLETE. UPDATE SUB-DOMAIN FACES NOW
        updateFace(lhs);

        // UPDATE BLACK CELLS
        // 1, 0, 0 CONFIGURATION
#pragma omp parallel for num_threads(inputParams.nThreads) default(none)
        for (int i = 1; i <= xEnd(vLevel); i+=2) {
            for (int j = 0; j <= yEnd(vLevel); j+=2) {
                for (int k = 0; k <= zEnd(vLevel); k+=2) {
                    lhs(vLevel)(i, j, k) = (1.0 - sorParam) * lhs(vLevel)(i, j, k) +
                                           (xix2(vLevel)(i) * ihx2(vLevel) * (lhs(vLevel)(i + 1, j, k) + lhs(vLevel)(i - 1, j, k)) +
                                            xixx(vLevel)(i) * i2hx(vLevel) * (lhs(vLevel)(i + 1, j, k) - lhs(vLevel)(i - 1, j, k)) +
                                            ety2(vLevel)(j) * ihy2(vLevel) * (lhs(vLevel)(i, j + 1, k) + lhs(vLevel)(i, j - 1, k)) +
                                            etyy(vLevel)(j) * i2hy(vLevel) * (lhs(vLevel)(i, j + 1, k) - lhs(vLevel)(i, j - 1, k)) +
                                            ztz2(vLevel)(k) * ihz2(vLevel) * (lhs(vLevel)(i, j, k + 1) + lhs(vLevel)(i, j, k - 1)) +
                                            ztzz(vLevel)(k) * i2hz(vLevel) * (lhs(vLevel)(i, j, k + 1) - lhs(vLevel)(i, j, k - 1)) -
                                                rhs(vLevel)(i, j, k)) * sorParam / (2.0 * (ihx2(vLevel)*xix2(vLevel)(i) + ihy2(vLevel)*ety2(vLevel)(j) + ihz2(vLevel)*ztz2(vLevel)(k)));
                }
            }
        }

        // 0, 1, 0 CONFIGURATION
#pragma omp parallel for num_threads(inputParams.nThreads) default(none)
        for (int i = 0; i <= xEnd(vLevel); i+=2) {
            for (int j = 1; j <= yEnd(vLevel); j+=2) {
                for (int k = 0; k <= zEnd(vLevel); k+=2) {
                    lhs(vLevel)(i, j, k) = (1.0 - sorParam) * lhs(vLevel)(i, j, k) +
                                           (xix2(vLevel)(i) * ihx2(vLevel) * (lhs(vLevel)(i + 1, j, k) + lhs(vLevel)(i - 1, j, k)) +
                                            xixx(vLevel)(i) * i2hx(vLevel) * (lhs(vLevel)(i + 1, j, k) - lhs(vLevel)(i - 1, j, k)) +
                                            ety2(vLevel)(j) * ihy2(vLevel) * (lhs(vLevel)(i, j + 1, k) + lhs(vLevel)(i, j - 1, k)) +
                                            etyy(vLevel)(j) * i2hy(vLevel) * (lhs(vLevel)(i, j + 1, k) - lhs(vLevel)(i, j - 1, k)) +
                                            ztz2(vLevel)(k) * ihz2(vLevel) * (lhs(vLevel)(i, j, k + 1) + lhs(vLevel)(i, j, k - 1)) +
                                            ztzz(vLevel)(k) * i2hz(vLevel) * (lhs(vLevel)(i, j, k + 1) - lhs(vLevel)(i, j, k - 1)) -
                                                rhs(vLevel)(i, j, k)) * sorParam / (2.0 * (ihx2(vLevel)*xix2(vLevel)(i) + ihy2(vLevel)*ety2(vLevel)(j) + ihz2(vLevel)*ztz2(vLevel)(k)));
                }
            }
        }

        // 0, 0, 1 CONFIGURATION
#pragma omp parallel for num_threads(inputParams.nThreads) default(none)
        for (int i = 0; i <= xEnd(vLevel); i+=2) {
            for (int j = 0; j <= yEnd(vLevel); j+=2) {
                for (int k = 1; k <= zEnd(vLevel); k+=2) {
                    lhs(vLevel)(i, j, k) = (1.0 - sorParam) * lhs(vLevel)(i, j, k) +
                                           (xix2(vLevel)(i) * ihx2(vLevel) * (lhs(vLevel)(i + 1, j, k) + lhs(vLevel)(i - 1, j, k)) +
                                            xixx(vLevel)(i) * i2hx(vLevel) * (lhs(vLevel)(i + 1, j, k) - lhs(vLevel)(i - 1, j, k)) +
                                            ety2(vLevel)(j) * ihy2(vLevel) * (lhs(vLevel)(i, j + 1, k) + lhs(vLevel)(i, j - 1, k)) +
                                            etyy(vLevel)(j) * i2hy(vLevel) * (lhs(vLevel)(i, j + 1, k) - lhs(vLevel)(i, j - 1, k)) +
                                            ztz2(vLevel)(k) * ihz2(vLevel) * (lhs(vLevel)(i, j, k + 1) + lhs(vLevel)(i, j, k - 1)) +
                                            ztzz(vLevel)(k) * i2hz(vLevel) * (lhs(vLevel)(i, j, k + 1) - lhs(vLevel)(i, j, k - 1)) -
                                                rhs(vLevel)(i, j, k)) * sorParam / (2.0 * (ihx2(vLevel)*xix2(vLevel)(i) + ihy2(vLevel)*ety2(vLevel)(j) + ihz2(vLevel)*ztz2(vLevel)(k)));
                }
            }
        }

        // 1, 1, 1 CONFIGURATION
#pragma omp parallel for num_threads(inputParams.nThreads) default(none)
        for (int i = 1; i <= xEnd(vLevel); i+=2) {
            for (int j = 1; j <= yEnd(vLevel); j+=2) {
                for (int k = 1; k <= zEnd(vLevel); k+=2) {
                    lhs(vLevel)(i, j, k) = (1.0 - sorParam) * lhs(vLevel)(i, j, k) +
                                           (xix2(vLevel)(i) * ihx2(vLevel) * (lhs(vLevel)(i + 1, j, k) + lhs(vLevel)(i - 1, j, k)) +
                                            xixx(vLevel)(i) * i2hx(vLevel) * (lhs(vLevel)(i + 1, j, k) - lhs(vLevel)(i - 1, j, k)) +
                                            ety2(vLevel)(j) * ihy2(vLevel) * (lhs(vLevel)(i, j + 1, k) + lhs(vLevel)(i, j - 1, k)) +
                                            etyy(vLevel)(j) * i2hy(vLevel) * (lhs(vLevel)(i, j + 1, k) - lhs(vLevel)(i, j - 1, k)) +
                                            ztz2(vLevel)(k) * ihz2(vLevel) * (lhs(vLevel)(i, j, k + 1) + lhs(vLevel)(i, j, k - 1)) +
                                            ztzz(vLevel)(k) * i2hz(vLevel) * (lhs(vLevel)(i, j, k + 1) - lhs(vLevel)(i, j, k - 1)) -
                                                rhs(vLevel)(i, j, k)) * sorParam / (2.0 * (ihx2(vLevel)*xix2(vLevel)(i) + ihy2(vLevel)*ety2(vLevel)(j) + ihz2(vLevel)*ztz2(vLevel)(k)));
                }
            }
        }

        tempValue = 0.0;
        localMax = -1.0e-10;
#pragma omp parallel for num_threads(inputParams.nThreads) default(none) private(tempValue) reduction(max: localMax)
        for (int i = 0; i <= xEnd(vLevel); ++i) {
            for (int j = 0; j <= yEnd(vLevel); ++j) {
                for (int k = 0; k <= zEnd(vLevel); ++k) {
                    tempValue = fabs(rhs(vLevel)(i, j, k) -
                               (xix2(vLevel)(i) * ihx2(vLevel) * (lhs(vLevel)(i + 1, j, k) - 2.0*lhs(vLevel)(i, j, k) + lhs(vLevel)(i - 1, j, k)) +
                                xixx(vLevel)(i) * i2hx(vLevel) * (lhs(vLevel)(i + 1, j, k) - lhs(vLevel)(i - 1, j, k)) +
                                ety2(vLevel)(j) * ihy2(vLevel) * (lhs(vLevel)(i, j + 1, k) - 2.0*lhs(vLevel)(i, j, k) + lhs(vLevel)(i, j - 1, k)) +
                                etyy(vLevel)(j) * i2hy(vLevel) * (lhs(vLevel)(i, j + 1, k) - lhs(vLevel)(i, j - 1, k)) +
                                ztz2(vLevel)(k) * ihz2(vLevel) * (lhs(vLevel)(i, j, k + 1) - 2.0*lhs(vLevel)(i, j, k) + lhs(vLevel)(i, j, k - 1)) +
                                ztzz(vLevel)(k) * i2hz(vLevel) * (lhs(vLevel)(i, j, k + 1) - lhs(vLevel)(i, j, k - 1))));

                    if (tempValue > localMax)
                        localMax = tempValue;
                }
            }
        }

        MPI_Allreduce(&localMax, &globalMax, 1, MPI_FP_REAL, MPI_MAX, MPI_COMM_WORLD);

        if (globalMax < inputParams.mgTolerance) break;

        iterCount += 1;
        if (iterCount > maxCount) {
            if (inputParams.printResidual)
                if (mesh.pf)
                    std::cout << "WARNING: Iterations for solution at coarsest level not converging." << std::endl;

            break;
        }
    }

    imposeBC();
}


void multigrid_d3::coarsen() {
    int pLevel;
    int i2, j2, k2;

    pLevel = vLevel;
    vLevel += 1;

#pragma omp parallel for num_threads(inputParams.nThreads) default(none) shared(pLevel) private(i2) private(j2) private(k2)
    for (int i = 0; i <= xEnd(vLevel); ++i) {
        i2 = i*2;
        for (int j = 0; j <= yEnd(vLevel); ++j) {
            j2 = j*2;
            for (int k = 0; k <= zEnd(vLevel); ++k) {
                k2 = k*2;
                rhs(vLevel)(i, j, k) = (tmp(pLevel)(i2, j2, k2) + tmp(pLevel)(i2 + 1, j2 + 1, k2 + 1) +
                                        tmp(pLevel)(i2, j2 + 1, k2 + 1) + tmp(pLevel)(i2 + 1, j2, k2) +
                                        tmp(pLevel)(i2 + 1, j2, k2 + 1) + tmp(pLevel)(i2, j2 + 1, k2) +
                                        tmp(pLevel)(i2 + 1, j2 + 1, k2) + tmp(pLevel)(i2, j2, k2 + 1))/8;
            }
        }
    }
}


void multigrid_d3::prolong() {
    int pLevel;
    int i2, j2, k2;

    pLevel = vLevel;
    vLevel -= 1;

    lhs(vLevel) = 0.0;

#pragma omp parallel for num_threads(inputParams.nThreads) default(none) shared(pLevel) private(i2) private(j2) private(k2)
    for (int i = 0; i <= xEnd(vLevel); ++i) {
        i2 = i/2;
        for (int j = 0; j <= yEnd(vLevel); j++) {
            j2 = j/2;
            for (int k = 0; k <= zEnd(vLevel); ++k) {
                k2 = k/2;
                lhs(vLevel)(i, j, k) = lhs(pLevel)(i2, j2, k2);
            }
        }
    }
}


real multigrid_d3::computeError(const int normOrder) {
    int pointCount = mesh.totalPoints;

    real residualVal = 0.0;
    real numValLoc = 0.0;
    real denValLoc = 0.0;
    real numValGlo = 0.0;
    real denValGlo = 0.0;
    real tempNum = 0.0;

    // This function is called at the finest grid level only.

    switch (normOrder) {
        case 0:     // L-Infinity Norm
#pragma omp parallel for num_threads(inputParams.nThreads) default(none) private(tempNum) reduction(max: numValLoc)
            for (int i = 0; i <= xEnd(0); ++i) {
                for (int j = 0; j <= yEnd(0); ++j) {
                    for (int k = 0; k <= zEnd(0); ++k) {
                        tempNum = fabs(rhs(0)(i, j, k) -
                                (xix2(0)(i) * ihx2(0) * (lhs(0)(i + 1, j, k) - 2.0*lhs(0)(i, j, k) + lhs(0)(i - 1, j, k)) +
                                 xixx(0)(i) * i2hx(0) * (lhs(0)(i + 1, j, k) - lhs(0)(i - 1, j, k)) +
                                 ety2(0)(j) * ihy2(0) * (lhs(0)(i, j + 1, k) - 2.0*lhs(0)(i, j, k) + lhs(0)(i, j - 1, k)) +
                                 etyy(0)(j) * i2hy(0) * (lhs(0)(i, j + 1, k) - lhs(0)(i, j - 1, k)) +
                                 ztz2(0)(k) * ihz2(0) * (lhs(0)(i, j, k + 1) - 2.0*lhs(0)(i, j, k) + lhs(0)(i, j, k - 1)) +
                                 ztzz(0)(k) * i2hz(0) * (lhs(0)(i, j, k + 1) - lhs(0)(i, j, k - 1))));

                        if (tempNum > numValLoc) numValLoc = tempNum;
                    }
                }
            }

            denValLoc = blitz::max(fabs(rhs(0)(stagCore(0))));
            MPI_Allreduce(&numValLoc, &numValGlo, 1, MPI_FP_REAL, MPI_MAX, MPI_COMM_WORLD);
            MPI_Allreduce(&denValLoc, &denValGlo, 1, MPI_FP_REAL, MPI_MAX, MPI_COMM_WORLD);

            if (denValGlo) {
                residualVal = numValGlo/denValGlo;
            } else {
                residualVal = numValGlo;
            }
            break;
        case 1:     // L-1 Norm
#pragma omp parallel for num_threads(inputParams.nThreads) default(none) private(tempNum) reduction(+: numValLoc)
            for (int i = 0; i <= xEnd(0); ++i) {
                for (int j = 0; j <= yEnd(0); ++j) {
                    for (int k = 0; k <= zEnd(0); ++k) {
                        tempNum = fabs(rhs(0)(i, j, k) -
                                (xix2(0)(i) * ihx2(0) * (lhs(0)(i + 1, j, k) - 2.0*lhs(0)(i, j, k) + lhs(0)(i - 1, j, k)) +
                                 xixx(0)(i) * i2hx(0) * (lhs(0)(i + 1, j, k) - lhs(0)(i - 1, j, k)) +
                                 ety2(0)(j) * ihy2(0) * (lhs(0)(i, j + 1, k) - 2.0*lhs(0)(i, j, k) + lhs(0)(i, j - 1, k)) +
                                 etyy(0)(j) * i2hy(0) * (lhs(0)(i, j + 1, k) - lhs(0)(i, j - 1, k)) +
                                 ztz2(0)(k) * ihz2(0) * (lhs(0)(i, j, k + 1) - 2.0*lhs(0)(i, j, k) + lhs(0)(i, j, k - 1)) +
                                 ztzz(0)(k) * i2hz(0) * (lhs(0)(i, j, k + 1) - lhs(0)(i, j, k - 1))));

                        numValLoc += tempNum;
                    }
                }
            }

            denValLoc = blitz::sum(fabs(rhs(0)(stagCore(0))));
            MPI_Allreduce(&numValLoc, &numValGlo, 1, MPI_FP_REAL, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&denValLoc, &denValGlo, 1, MPI_FP_REAL, MPI_SUM, MPI_COMM_WORLD);

            if (denValGlo) {
                residualVal = numValGlo/denValGlo;
            } else {
                residualVal = numValGlo/pointCount;
            }
            break;
        case 2:     // L-2 Norm
#pragma omp parallel for num_threads(inputParams.nThreads) default(none) private(tempNum) reduction(+: numValLoc)
            for (int i = 0; i <= xEnd(0); ++i) {
                for (int j = 0; j <= yEnd(0); ++j) {
                    for (int k = 0; k <= zEnd(0); ++k) {
                        tempNum = rhs(0)(i, j, k) -
                                   (xix2(0)(i) * ihx2(0) * (lhs(0)(i + 1, j, k) - 2.0*lhs(0)(i, j, k) + lhs(0)(i - 1, j, k)) +
                                    xixx(0)(i) * i2hx(0) * (lhs(0)(i + 1, j, k) - lhs(0)(i - 1, j, k)) +
                                    ety2(0)(j) * ihy2(0) * (lhs(0)(i, j + 1, k) - 2.0*lhs(0)(i, j, k) + lhs(0)(i, j - 1, k)) +
                                    etyy(0)(j) * i2hy(0) * (lhs(0)(i, j + 1, k) - lhs(0)(i, j - 1, k)) +
                                    ztz2(0)(k) * ihz2(0) * (lhs(0)(i, j, k + 1) - 2.0*lhs(0)(i, j, k) + lhs(0)(i, j, k - 1)) +
                                    ztzz(0)(k) * i2hz(0) * (lhs(0)(i, j, k + 1) - lhs(0)(i, j, k - 1)));

                        numValLoc += tempNum*tempNum;
                    }
                }
            }

            denValLoc = blitz::sum(blitz::pow(rhs(0)(stagCore(0)), 2));
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


void multigrid_d3::createMGSubArrays() {
    int count, length, stride;

    recvStatus.resize(12);
    recvRequest.resize(12);

    xFace.resize(mesh.vcDepth + 1);
    yFace.resize(mesh.vcDepth + 1);
    zFace.resize(mesh.vcDepth + 1);

    xEdge.resize(mesh.vcDepth + 1);
    yEdge.resize(mesh.vcDepth + 1);
    zEdge.resize(mesh.vcDepth + 1);

    sendInd.resize(mesh.vcDepth + 1, 26);        recvInd.resize(mesh.vcDepth + 1, 26);

    for(int n=0; n<=mesh.vcDepth; n++) {
        int xCount = stagFull(n).ubound(0) + 2;
        int yCount = stagFull(n).ubound(1) + 2;
        int zCount = stagFull(n).ubound(2) + 2;

        /*************************** MPI DATATYPES FOR FACE TRANSFER ***************************/

        ////// ALONG X-DIRECTION //////
        count = yCount*zCount;

        MPI_Type_contiguous(count, MPI_FP_REAL, &xFace(n));
        MPI_Type_commit(&xFace(n));

        ////// ALONG Y-DIRECTION //////
        count = xCount;
        length = zCount;
        stride = zCount*yCount;

        MPI_Type_vector(count, length, stride, MPI_FP_REAL, &yFace(n));
        MPI_Type_commit(&yFace(n));

        ////// ALONG Z-DIRECTION //////
        count = xCount*yCount;
        length = 1;
        stride = zCount;

        MPI_Type_vector(count, length, stride, MPI_FP_REAL, &zFace(n));
        MPI_Type_commit(&zFace(n));

        /*************************** MPI DATATYPES FOR EDGE TRANSFER ***************************/

        ////// ALONG XY-PLANE //////
        count = zCount;

        MPI_Type_contiguous(count, MPI_FP_REAL, &zEdge(n));
        MPI_Type_commit(&zEdge(n));

        ////// ALONG YZ-PLANE //////
        count = xCount;
        length = 1;
        stride = yCount*zCount;

        MPI_Type_vector(count, length, stride, MPI_FP_REAL, &xEdge(n));
        MPI_Type_commit(&xEdge(n));

        ////// ALONG ZX-PLANE //////
        count = yCount;
        length = 1;
        stride = zCount;

        MPI_Type_vector(count, length, stride, MPI_FP_REAL, &yEdge(n));
        MPI_Type_commit(&yEdge(n));

        // SET STARTING INDICES OF MEMORY LOCATIONS FROM WHERE TO READ (SEND) AND WRITE (RECEIVE) DATA
        sendInd(n, 0) = 0, -1, -1;          sendInd(n, 1) = stagCore(n).ubound(0), -1, -1;
        sendInd(n, 2) = -1, 0, -1;          sendInd(n, 3) = -1, stagCore(n).ubound(1), -1;
        sendInd(n, 4) = -1, -1, 0;          sendInd(n, 5) = -1, -1, stagCore(n).ubound(2);

        sendInd(n, 6) = 0, 0, -1;       sendInd(n, 7) = 0, stagCore(n).ubound(1), -1;       sendInd(n, 8) = stagCore(n).ubound(0), 0, -1;       sendInd(n, 9) = stagCore(n).ubound(0), stagCore(n).ubound(1), -1;
        sendInd(n, 10) = -1, 0, 0;      sendInd(n, 11) = -1, 0, stagCore(n).ubound(2);      sendInd(n, 12) = -1, stagCore(n).ubound(1), 0;      sendInd(n, 13) = -1, stagCore(n).ubound(1), stagCore(n).ubound(2);
        sendInd(n, 14) = 0, -1, 0;      sendInd(n, 15) = stagCore(n).ubound(0), -1, 0;      sendInd(n, 16) = 0, -1, stagCore(n).ubound(2);      sendInd(n, 17) = stagCore(n).ubound(0), -1, stagCore(n).ubound(2);

        sendInd(n, 18) = 0, 0, 0;
        sendInd(n, 19) = 0, stagCore(n).ubound(1), 0;
        sendInd(n, 20) = stagCore(n).ubound(0), 0, 0;
        sendInd(n, 21) = stagCore(n).ubound(0), stagCore(n).ubound(1), 0;

        sendInd(n, 22) = 0, 0, stagCore(n).ubound(2);
        sendInd(n, 23) = 0, stagCore(n).ubound(1), stagCore(n).ubound(2);
        sendInd(n, 24) = stagCore(n).ubound(0), 0, stagCore(n).ubound(2);
        sendInd(n, 25) = stagCore(n).ubound(0), stagCore(n).ubound(1), stagCore(n).ubound(2);


        recvInd(n, 0) = -1, -1, -1;         recvInd(n, 1) = stagCore(n).ubound(0) + 1, -1, -1;
        recvInd(n, 2) = -1, -1, -1;         recvInd(n, 3) = -1, stagCore(n).ubound(1) + 1, -1;
        recvInd(n, 4) = -1, -1, -1;         recvInd(n, 5) = -1, -1, stagCore(n).ubound(2) + 1;

        recvInd(n, 6) = -1, -1, -1;     recvInd(n, 7) = -1, stagCore(n).ubound(1) + 1, -1;      recvInd(n, 8) = stagCore(n).ubound(0) + 1, -1, -1;       recvInd(n, 9) = stagCore(n).ubound(0) + 1, stagCore(n).ubound(1) + 1, -1;
        recvInd(n, 10) = -1, -1, -1;    recvInd(n, 11) = -1, -1, stagCore(n).ubound(2) + 1;     recvInd(n, 12) = -1, stagCore(n).ubound(1) + 1, -1;      recvInd(n, 13) = -1, stagCore(n).ubound(1) + 1, stagCore(n).ubound(2) + 1;
        recvInd(n, 14) = -1, -1, -1;    recvInd(n, 15) = stagCore(n).ubound(0) + 1, -1, -1;     recvInd(n, 16) = -1, -1, stagCore(n).ubound(2) + 1;      recvInd(n, 17) = stagCore(n).ubound(0) + 1, -1, stagCore(n).ubound(2) + 1;

        recvInd(n, 18) = -1, -1, -1;
        recvInd(n, 19) = -1, stagCore(n).ubound(1) + 1, -1;
        recvInd(n, 20) = stagCore(n).ubound(0) + 1, -1, -1;
        recvInd(n, 21) = stagCore(n).ubound(0) + 1, stagCore(n).ubound(1) + 1, -1;

        recvInd(n, 22) = -1, -1, stagCore(n).ubound(2) + 1;
        recvInd(n, 23) = -1, stagCore(n).ubound(1) + 1, stagCore(n).ubound(2) + 1;
        recvInd(n, 24) = stagCore(n).ubound(0) + 1, -1, stagCore(n).ubound(2) + 1;
        recvInd(n, 25) = stagCore(n).ubound(0) + 1, stagCore(n).ubound(1) + 1, stagCore(n).ubound(2) + 1;
    }
}


void multigrid_d3::initDirichlet() {
    real xDist, yDist, zDist;

    // Generate the walls as 2D Blitz arrays
    xWall.resize(stagFull(0).ubound(1) - stagFull(0).lbound(1) + 1, stagFull(0).ubound(2) - stagFull(0).lbound(2) + 1);
    xWall.reindexSelf(blitz::TinyVector<int, 2>(stagFull(0).lbound(1), stagFull(0).lbound(2)));
    xWall = 0.0;

    yWall.resize(stagFull(0).ubound(0) - stagFull(0).lbound(0) + 1, stagFull(0).ubound(2) - stagFull(0).lbound(2) + 1);
    yWall.reindexSelf(blitz::TinyVector<int, 2>(stagFull(0).lbound(0), stagFull(0).lbound(2)));
    yWall = 0.0;

    zWall.resize(stagFull(0).ubound(0) - stagFull(0).lbound(0) + 1, stagFull(0).ubound(1) - stagFull(0).lbound(1) + 1);
    zWall.reindexSelf(blitz::TinyVector<int, 2>(stagFull(0).lbound(0), stagFull(0).lbound(1)));
    zWall = 0.0;

    // Compute values at the walls using the (r^2)/6 formula
    // Along X-direction - Left and Right Walls
    xDist = mesh.inputParams.Lx/2.0;
    for (int j=0; j<=stagCore(0).ubound(1); ++j) {
        yDist = mesh.y(j) - 0.5;
        for (int k=0; k<=stagCore(0).ubound(2); ++k) {
            zDist = mesh.z(k) - 0.5;

            xWall(j, k) = (xDist*xDist + yDist*yDist + zDist*zDist)/6.0;
        }
    }

    // Along Y-direction - Front and Rear Walls
    yDist = mesh.inputParams.Ly/2.0;
    for (int i=0; i<=stagCore(0).ubound(0); ++i) {
        xDist = mesh.x(i) - 0.5;
        for (int k=0; k<=stagCore(0).ubound(2); ++k) {
            zDist = mesh.z(k) - 0.5;

            yWall(i, k) = (xDist*xDist + yDist*yDist + zDist*zDist)/6.0;
        }
    }

    // Along Z-direction - Top and Bottom Walls
    zDist = mesh.inputParams.Lz/2.0;
    for (int i=0; i<=stagCore(0).ubound(0); ++i) {
        xDist = mesh.x(i) - 0.5;
        for (int j=0; j<=stagCore(0).ubound(1); ++j) {
            yDist = mesh.y(j) - 0.5;

            zWall(i, j) = (xDist*xDist + yDist*yDist + zDist*zDist)/6.0;
        }
    }
}


void multigrid_d3::imposeBC() {
    // FOR PARALLEL RUNS, FIRST UPDATE GHOST POINTS OF MPI SUB-DOMAINS
    updateFace(lhs);

    if (not inputParams.xPer) {
#ifdef TEST_POISSON
        // DIRICHLET BOUNDARY CONDITION AT LEFT AND RIGHT WALLS
        if (zeroBC) {
            if (xfr) lhs(vLevel)(-1, all, all) = -lhs(vLevel)(0, all, all);
            if (xlr) lhs(vLevel)(stagCore(vLevel).ubound(0) + 1, all, all) = -lhs(vLevel)(stagCore(vLevel).ubound(0), all, all);

        } else {
            if (xfr) lhs(vLevel)(-1, all, all) = 2.0*xWall(all, all) - lhs(vLevel)(0, all, all);
            if (xlr) lhs(vLevel)(stagCore(vLevel).ubound(0) + 1, all, all) = 2.0*xWall(all, all) - lhs(vLevel)(stagCore(vLevel).ubound(0), all, all);
        }
#else
        if (allNeumann) {
            // NEUMANN BOUNDARY CONDITION AT LEFT AND RIGHT WALLS
            if (xfr) lhs(vLevel)(-1, all, all) = lhs(vLevel)(0, all, all);
            if (xlr) lhs(vLevel)(stagCore(vLevel).ubound(0) + 1, all, all) = lhs(vLevel)(stagCore(vLevel).ubound(0), all, all);
        } else {
            // DIRICHLET BOUNDARY CONDITION AT LEFT AND RIGHT WALLS
            if (xfr) lhs(vLevel)(-1, all, all) = -lhs(vLevel)(0, all, all);
            if (xlr) lhs(vLevel)(stagCore(vLevel).ubound(0) + 1, all, all) = -lhs(vLevel)(stagCore(vLevel).ubound(0), all, all);
        }
#endif
    } // PERIODIC BOUNDARY CONDITIONS ARE AUTOMATICALLY IMPOSED BY PERIODIC DATA TRANSFER ACROSS PROCESSORS THROUGH updateFace()


    if (not inputParams.yPer) {
#ifdef TEST_POISSON
        // DIRICHLET BOUNDARY CONDITION AT FRONT AND BACK WALLS
        if (zeroBC) {
            if (yfr) lhs(vLevel)(all, -1, all) = -lhs(vLevel)(all, 0, all);
            if (ylr) lhs(vLevel)(all, stagCore(vLevel).ubound(1) + 1, all) = -lhs(vLevel)(all, stagCore(vLevel).ubound(1), all);

        } else {
            if (yfr) lhs(vLevel)(all, -1, all) = 2.0*yWall(all, all) - lhs(vLevel)(all, 0, all);
            if (ylr) lhs(vLevel)(all, stagCore(vLevel).ubound(1) + 1, all) = 2.0*yWall(all, all) - lhs(vLevel)(all, stagCore(vLevel).ubound(1), all);
        }
#else
        if (allNeumann) {
            // NEUMANN BOUNDARY CONDITION AT FRONT AND BACK WALLS
            if (yfr) lhs(vLevel)(all, -1, all) = lhs(vLevel)(all, 0, all);
            if (ylr) lhs(vLevel)(all, stagCore(vLevel).ubound(1) + 1, all) = lhs(vLevel)(all, stagCore(vLevel).ubound(1), all);
        } else {
            // DIRICHLET BOUNDARY CONDITION AT FRONT AND BACK WALLS
            if (yfr) lhs(vLevel)(all, -1, all) = -lhs(vLevel)(all, 0, all);
            if (ylr) lhs(vLevel)(all, stagCore(vLevel).ubound(1) + 1, all) = -lhs(vLevel)(all, stagCore(vLevel).ubound(1), all);
        }
#endif
    } // PERIODIC BOUNDARY CONDITIONS ARE AUTOMATICALLY IMPOSED BY PERIODIC DATA TRANSFER ACROSS PROCESSORS THROUGH updateFace()


    if (not inputParams.zPer) {
#ifdef TEST_POISSON
        // DIRICHLET BOUNDARY CONDITION AT BOTTOM AND TOP WALLS
        if (zeroBC) {
            if (zfr) lhs(vLevel)(all, all, -1) = -lhs(vLevel)(all, all, 0);
            if (zlr) lhs(vLevel)(all, all, stagCore(vLevel).ubound(2) + 1) = -lhs(vLevel)(all, all, stagCore(vLevel).ubound(2));
        } else {
            if (zfr) lhs(vLevel)(all, all, -1) = 2.0*zWall(all, all) - lhs(vLevel)(all, all, 0);
            if (zlr) lhs(vLevel)(all, all, stagCore(vLevel).ubound(2) + 1) = 2.0*zWall(all, all) - lhs(vLevel)(all, all, stagCore(vLevel).ubound(2));
        }
#else
        if (allNeumann) {
            // NEUMANN BOUNDARY CONDITION AT BOTTOM AND TOP WALLS
            if (zfr) lhs(vLevel)(all, all, -1) = lhs(vLevel)(all, all, 0);
            if (zlr) lhs(vLevel)(all, all, stagCore(vLevel).ubound(2) + 1) = lhs(vLevel)(all, all, stagCore(vLevel).ubound(2));
        } else {
            // DIRICHLET BOUNDARY CONDITION AT BOTTOM AND TOP WALLS
            if (zfr) lhs(vLevel)(all, all, -1) = -lhs(vLevel)(all, all, 0);
            if (zlr) lhs(vLevel)(all, all, stagCore(vLevel).ubound(2) + 1) = -lhs(vLevel)(all, all, stagCore(vLevel).ubound(2));
        }
#endif
    } // PERIODIC BOUNDARY CONDITIONS ARE AUTOMATICALLY IMPOSED BY PERIODIC DATA TRANSFER ACROSS PROCESSORS THROUGH updateFace()
}


void multigrid_d3::updateFace(blitz::Array<blitz::Array<real, 3>, 1> &data) {
    recvRequest = MPI_REQUEST_NULL;

    // TRANSFER DATA ACROSS FACES FROM NEIGHBOURING CELLS TO IMPOSE SUB-DOMAIN BOUNDARY CONDITIONS
    MPI_Irecv(&(data(vLevel)(recvInd(vLevel, 0))), 1, xFace(vLevel), mesh.rankData.faceRanks(0), 1, MPI_COMM_WORLD, &recvRequest(0));
    MPI_Irecv(&(data(vLevel)(recvInd(vLevel, 1))), 1, xFace(vLevel), mesh.rankData.faceRanks(1), 2, MPI_COMM_WORLD, &recvRequest(1));
    MPI_Irecv(&(data(vLevel)(recvInd(vLevel, 2))), 1, yFace(vLevel), mesh.rankData.faceRanks(2), 3, MPI_COMM_WORLD, &recvRequest(2));
    MPI_Irecv(&(data(vLevel)(recvInd(vLevel, 3))), 1, yFace(vLevel), mesh.rankData.faceRanks(3), 4, MPI_COMM_WORLD, &recvRequest(3));
    MPI_Irecv(&(data(vLevel)(recvInd(vLevel, 4))), 1, zFace(vLevel), mesh.rankData.faceRanks(4), 5, MPI_COMM_WORLD, &recvRequest(4));
    MPI_Irecv(&(data(vLevel)(recvInd(vLevel, 5))), 1, zFace(vLevel), mesh.rankData.faceRanks(5), 6, MPI_COMM_WORLD, &recvRequest(5));

    MPI_Send(&(data(vLevel)(sendInd(vLevel, 0))), 1, xFace(vLevel), mesh.rankData.faceRanks(0), 2, MPI_COMM_WORLD);
    MPI_Send(&(data(vLevel)(sendInd(vLevel, 1))), 1, xFace(vLevel), mesh.rankData.faceRanks(1), 1, MPI_COMM_WORLD);
    MPI_Send(&(data(vLevel)(sendInd(vLevel, 2))), 1, yFace(vLevel), mesh.rankData.faceRanks(2), 4, MPI_COMM_WORLD);
    MPI_Send(&(data(vLevel)(sendInd(vLevel, 3))), 1, yFace(vLevel), mesh.rankData.faceRanks(3), 3, MPI_COMM_WORLD);
    MPI_Send(&(data(vLevel)(sendInd(vLevel, 4))), 1, zFace(vLevel), mesh.rankData.faceRanks(4), 6, MPI_COMM_WORLD);
    MPI_Send(&(data(vLevel)(sendInd(vLevel, 5))), 1, zFace(vLevel), mesh.rankData.faceRanks(5), 5, MPI_COMM_WORLD);

    MPI_Waitall(6, recvRequest.dataFirst(), recvStatus.dataFirst());
}


void multigrid_d3::updateFull(blitz::Array<blitz::Array<real, 3>, 1> &data) {
    recvRequest = MPI_REQUEST_NULL;

    // TRANSFER DATA ACROSS FACES FROM NEIGHBOURING CELLS TO IMPOSE SUB-DOMAIN BOUNDARY CONDITIONS
    updateFace(data);

    // TRANSFER DATA ACROSS EDGES FROM NEIGHBOURING CELLS TO IMPOSE SUB-DOMAIN BOUNDARY CONDITIONS
    MPI_Irecv(&(data(vLevel)(recvInd(vLevel, 6))), 1, zEdge(vLevel), mesh.rankData.edgeRanks(0), 1, MPI_COMM_WORLD, &recvRequest(0));
    MPI_Irecv(&(data(vLevel)(recvInd(vLevel, 7))), 1, zEdge(vLevel), mesh.rankData.edgeRanks(1), 2, MPI_COMM_WORLD, &recvRequest(1));
    MPI_Irecv(&(data(vLevel)(recvInd(vLevel, 8))), 1, zEdge(vLevel), mesh.rankData.edgeRanks(2), 3, MPI_COMM_WORLD, &recvRequest(2));
    MPI_Irecv(&(data(vLevel)(recvInd(vLevel, 9))), 1, zEdge(vLevel), mesh.rankData.edgeRanks(3), 4, MPI_COMM_WORLD, &recvRequest(3));

    MPI_Irecv(&(data(vLevel)(recvInd(vLevel,10))), 1, xEdge(vLevel), mesh.rankData.edgeRanks(4), 5, MPI_COMM_WORLD, &recvRequest(4));
    MPI_Irecv(&(data(vLevel)(recvInd(vLevel,11))), 1, xEdge(vLevel), mesh.rankData.edgeRanks(5), 6, MPI_COMM_WORLD, &recvRequest(5));
    MPI_Irecv(&(data(vLevel)(recvInd(vLevel,12))), 1, xEdge(vLevel), mesh.rankData.edgeRanks(6), 7, MPI_COMM_WORLD, &recvRequest(6));
    MPI_Irecv(&(data(vLevel)(recvInd(vLevel,13))), 1, xEdge(vLevel), mesh.rankData.edgeRanks(7), 8, MPI_COMM_WORLD, &recvRequest(7));

    MPI_Irecv(&(data(vLevel)(recvInd(vLevel,14))), 1, yEdge(vLevel), mesh.rankData.edgeRanks(8),  9, MPI_COMM_WORLD, &recvRequest(8));
    MPI_Irecv(&(data(vLevel)(recvInd(vLevel,15))), 1, yEdge(vLevel), mesh.rankData.edgeRanks(9), 10, MPI_COMM_WORLD, &recvRequest(9));
    MPI_Irecv(&(data(vLevel)(recvInd(vLevel,16))), 1, yEdge(vLevel), mesh.rankData.edgeRanks(10),11, MPI_COMM_WORLD, &recvRequest(10));
    MPI_Irecv(&(data(vLevel)(recvInd(vLevel,17))), 1, yEdge(vLevel), mesh.rankData.edgeRanks(11),12, MPI_COMM_WORLD, &recvRequest(11));

    MPI_Send(&(data(vLevel)(sendInd(vLevel, 6))), 1, zEdge(vLevel), mesh.rankData.edgeRanks(0), 4, MPI_COMM_WORLD);
    MPI_Send(&(data(vLevel)(sendInd(vLevel, 7))), 1, zEdge(vLevel), mesh.rankData.edgeRanks(1), 3, MPI_COMM_WORLD);
    MPI_Send(&(data(vLevel)(sendInd(vLevel, 8))), 1, zEdge(vLevel), mesh.rankData.edgeRanks(2), 2, MPI_COMM_WORLD);
    MPI_Send(&(data(vLevel)(sendInd(vLevel, 9))), 1, zEdge(vLevel), mesh.rankData.edgeRanks(3), 1, MPI_COMM_WORLD);
                                                                                      
    MPI_Send(&(data(vLevel)(sendInd(vLevel,10))), 1, xEdge(vLevel), mesh.rankData.edgeRanks(4), 8, MPI_COMM_WORLD);
    MPI_Send(&(data(vLevel)(sendInd(vLevel,11))), 1, xEdge(vLevel), mesh.rankData.edgeRanks(5), 7, MPI_COMM_WORLD);
    MPI_Send(&(data(vLevel)(sendInd(vLevel,12))), 1, xEdge(vLevel), mesh.rankData.edgeRanks(6), 6, MPI_COMM_WORLD);
    MPI_Send(&(data(vLevel)(sendInd(vLevel,13))), 1, xEdge(vLevel), mesh.rankData.edgeRanks(7), 5, MPI_COMM_WORLD);
                                                                                             
    MPI_Send(&(data(vLevel)(sendInd(vLevel,14))), 1, yEdge(vLevel), mesh.rankData.edgeRanks(8), 12, MPI_COMM_WORLD);
    MPI_Send(&(data(vLevel)(sendInd(vLevel,15))), 1, yEdge(vLevel), mesh.rankData.edgeRanks(9), 11, MPI_COMM_WORLD);
    MPI_Send(&(data(vLevel)(sendInd(vLevel,16))), 1, yEdge(vLevel), mesh.rankData.edgeRanks(10),10, MPI_COMM_WORLD);
    MPI_Send(&(data(vLevel)(sendInd(vLevel,17))), 1, yEdge(vLevel), mesh.rankData.edgeRanks(11), 9, MPI_COMM_WORLD);

    MPI_Waitall(12, recvRequest.dataFirst(), recvStatus.dataFirst());

    // TRANSFER DATA ACROSS CORNERS FROM NEIGHBOURING CELLS TO IMPOSE SUB-DOMAIN BOUNDARY CONDITIONS
    MPI_Irecv(&(data(vLevel)(recvInd(vLevel,18))), 1, MPI_FP_REAL, mesh.rankData.cornRanks(0), 1, MPI_COMM_WORLD, &recvRequest(0));
    MPI_Irecv(&(data(vLevel)(recvInd(vLevel,19))), 1, MPI_FP_REAL, mesh.rankData.cornRanks(1), 2, MPI_COMM_WORLD, &recvRequest(1));
    MPI_Irecv(&(data(vLevel)(recvInd(vLevel,20))), 1, MPI_FP_REAL, mesh.rankData.cornRanks(2), 3, MPI_COMM_WORLD, &recvRequest(2));
    MPI_Irecv(&(data(vLevel)(recvInd(vLevel,21))), 1, MPI_FP_REAL, mesh.rankData.cornRanks(3), 4, MPI_COMM_WORLD, &recvRequest(3));
    MPI_Irecv(&(data(vLevel)(recvInd(vLevel,22))), 1, MPI_FP_REAL, mesh.rankData.cornRanks(4), 5, MPI_COMM_WORLD, &recvRequest(4));
    MPI_Irecv(&(data(vLevel)(recvInd(vLevel,23))), 1, MPI_FP_REAL, mesh.rankData.cornRanks(5), 6, MPI_COMM_WORLD, &recvRequest(5));
    MPI_Irecv(&(data(vLevel)(recvInd(vLevel,24))), 1, MPI_FP_REAL, mesh.rankData.cornRanks(6), 7, MPI_COMM_WORLD, &recvRequest(6));
    MPI_Irecv(&(data(vLevel)(recvInd(vLevel,25))), 1, MPI_FP_REAL, mesh.rankData.cornRanks(7), 8, MPI_COMM_WORLD, &recvRequest(7));

    MPI_Send(&(data(vLevel)(sendInd(vLevel,18))), 1, MPI_FP_REAL, mesh.rankData.cornRanks(0), 8, MPI_COMM_WORLD);
    MPI_Send(&(data(vLevel)(sendInd(vLevel,19))), 1, MPI_FP_REAL, mesh.rankData.cornRanks(1), 7, MPI_COMM_WORLD);
    MPI_Send(&(data(vLevel)(sendInd(vLevel,20))), 1, MPI_FP_REAL, mesh.rankData.cornRanks(2), 6, MPI_COMM_WORLD);
    MPI_Send(&(data(vLevel)(sendInd(vLevel,21))), 1, MPI_FP_REAL, mesh.rankData.cornRanks(3), 5, MPI_COMM_WORLD);
    MPI_Send(&(data(vLevel)(sendInd(vLevel,22))), 1, MPI_FP_REAL, mesh.rankData.cornRanks(4), 4, MPI_COMM_WORLD);
    MPI_Send(&(data(vLevel)(sendInd(vLevel,23))), 1, MPI_FP_REAL, mesh.rankData.cornRanks(5), 3, MPI_COMM_WORLD);
    MPI_Send(&(data(vLevel)(sendInd(vLevel,24))), 1, MPI_FP_REAL, mesh.rankData.cornRanks(6), 2, MPI_COMM_WORLD);
    MPI_Send(&(data(vLevel)(sendInd(vLevel,25))), 1, MPI_FP_REAL, mesh.rankData.cornRanks(7), 1, MPI_COMM_WORLD);

    MPI_Waitall(8, recvRequest.dataFirst(), recvStatus.dataFirst());
}

