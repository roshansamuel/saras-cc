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
/*! \file predictors.cc
 *
 *  \brief Definitions for iterative solvers of class timestep
 *  \sa timestep.h
 *  \author Roshan Samuel
 *  \date Nov 2019
 *  \copyright New BSD License
 *
 ********************************************************************************************************************************************
 */

#include "timestep.h"


/**
 ********************************************************************************************************************************************
 * \brief   Function to solve the implicit equation for x-velocity
 *
 *          The implicit equation for \f$ u_x' \f$ of the implicit Crank-Nicholson method is solved using the Jacobi
 *          iterative method here.
 *
 *          The loop exits when the global maximum of the error in computed solution falls below the specified tolerance.
 *          If the solution doesn't converge even after an internally assigned maximum number for iterations, the solver
 *          aborts with an error message.
 *
 ********************************************************************************************************************************************
 */
void timestep::solveVx(vfield &V, plainvf &nseRHS, real beta) {
    int iterCount = 0;
    real locMax = 0.0;
    real gloMax = 0.0;

    while (true) {
#ifdef PLANAR
        int iY = 0;
#pragma omp parallel for num_threads(mesh.inputParams.nThreads) default(none) shared(V) shared(nseRHS) shared(iY) shared(beta)
        for (int iX = xSt; iX <= xEn; iX++) {
            for (int iZ = zSt; iZ <= zEn; iZ++) {
                iterTemp(iX, iY, iZ) = ((ihx2 * mesh.xix2(iX) * (V.Vx.F(iX+1, iY, iZ) + V.Vx.F(iX-1, iY, iZ)) +
                                       i2hx * mesh.xixx(iX) * (V.Vx.F(iX+1, iY, iZ) - V.Vx.F(iX-1, iY, iZ)) +
                                       ihz2 * mesh.ztz2(iZ) * (V.Vx.F(iX, iY, iZ+1) + V.Vx.F(iX, iY, iZ-1)) +
                                       i2hz * mesh.ztzz(iZ) * (V.Vx.F(iX, iY, iZ+1) - V.Vx.F(iX, iY, iZ-1))) *
                        dt * nu * beta + nseRHS.Vx(iX, iY, iZ)) /
           (1.0 + 2.0 * dt * nu * beta * (ihx2 * mesh.xix2(iX) + ihz2 * mesh.ztz2(iZ)));
            }
        }
#else
#pragma omp parallel for num_threads(mesh.inputParams.nThreads) default(none) shared(V) shared(nseRHS) shared(beta)
        for (int iX = xSt; iX <= xEn; iX++) {
            for (int iY = ySt; iY <= yEn; iY++) {
                for (int iZ = zSt; iZ <= zEn; iZ++) {
                    iterTemp(iX, iY, iZ) = ((ihx2 * mesh.xix2(iX) * (V.Vx.F(iX+1, iY, iZ) + V.Vx.F(iX-1, iY, iZ)) +
                                             i2hx * mesh.xixx(iX) * (V.Vx.F(iX+1, iY, iZ) - V.Vx.F(iX-1, iY, iZ)) +
                                             ihy2 * mesh.ety2(iY) * (V.Vx.F(iX, iY+1, iZ) + V.Vx.F(iX, iY-1, iZ)) +
                                             i2hy * mesh.etyy(iY) * (V.Vx.F(iX, iY+1, iZ) - V.Vx.F(iX, iY-1, iZ)) +
                                             ihz2 * mesh.ztz2(iZ) * (V.Vx.F(iX, iY, iZ+1) + V.Vx.F(iX, iY, iZ-1)) +
                                             i2hz * mesh.ztzz(iZ) * (V.Vx.F(iX, iY, iZ+1) - V.Vx.F(iX, iY, iZ-1))) *
                            dt * nu * beta + nseRHS.Vx(iX, iY, iZ)) /
               (1.0 + 2.0 * dt * nu * beta * (ihx2 * mesh.xix2(iX) + ihy2 * mesh.ety2(iY) + ihz2 * mesh.ztz2(iZ)));
                }
            }
        }
#endif

        V.Vx.F = iterTemp;

        V.imposeVxBC();

#ifdef PLANAR
#pragma omp parallel for num_threads(mesh.inputParams.nThreads) default(none) shared(V) shared(iY) shared(beta)
        for (int iX = xSt; iX <= xEn; iX++) {
            for (int iZ = zSt; iZ <= zEn; iZ++) {
                iterTemp(iX, iY, iZ) = V.Vx.F(iX, iY, iZ) - beta * dt * nu * (
                          mesh.xix2(iX) * (V.Vx.F(iX+1, iY, iZ) - 2.0 * V.Vx.F(iX, iY, iZ) + V.Vx.F(iX-1, iY, iZ)) * ihx2 +
                          mesh.xixx(iX) * (V.Vx.F(iX+1, iY, iZ) - V.Vx.F(iX-1, iY, iZ)) * i2hx +
                          mesh.ztz2(iZ) * (V.Vx.F(iX, iY, iZ+1) - 2.0 * V.Vx.F(iX, iY, iZ) + V.Vx.F(iX, iY, iZ-1)) * ihz2 +
                          mesh.ztzz(iZ) * (V.Vx.F(iX, iY, iZ+1) - V.Vx.F(iX, iY, iZ-1)) * i2hz);
            }
        }
#else
#pragma omp parallel for num_threads(mesh.inputParams.nThreads) default(none) shared(V) shared(beta)
        for (int iX = xSt; iX <= xEn; iX++) {
            for (int iY = ySt; iY <= yEn; iY++) {
                for (int iZ = zSt; iZ <= zEn; iZ++) {
                    iterTemp(iX, iY, iZ) = V.Vx.F(iX, iY, iZ) - beta * dt * nu * (
                              mesh.xix2(iX) * (V.Vx.F(iX+1, iY, iZ) - 2.0 * V.Vx.F(iX, iY, iZ) + V.Vx.F(iX-1, iY, iZ)) * ihx2 +
                              mesh.xixx(iX) * (V.Vx.F(iX+1, iY, iZ) - V.Vx.F(iX-1, iY, iZ)) * i2hx +
                              mesh.ety2(iY) * (V.Vx.F(iX, iY+1, iZ) - 2.0 * V.Vx.F(iX, iY, iZ) + V.Vx.F(iX, iY-1, iZ)) * ihy2 +
                              mesh.etyy(iY) * (V.Vx.F(iX, iY+1, iZ) - V.Vx.F(iX, iY-1, iZ)) * i2hy +
                              mesh.ztz2(iZ) * (V.Vx.F(iX, iY, iZ+1) - 2.0 * V.Vx.F(iX, iY, iZ) + V.Vx.F(iX, iY, iZ-1)) * ihz2 +
                              mesh.ztzz(iZ) * (V.Vx.F(iX, iY, iZ+1) - V.Vx.F(iX, iY, iZ-1)) * i2hz);
                }
            }
        }
#endif

        iterTemp(core) = abs(iterTemp(core) - nseRHS.Vx(core));

        locMax = blitz::max(iterTemp(core));
        MPI_Allreduce(&locMax, &gloMax, 1, MPI_FP_REAL, MPI_MAX, MPI_COMM_WORLD);

        if (gloMax < mesh.inputParams.cnTolerance) break;

        iterCount += 1;

        if (iterCount > maxIterations) {
            if (mesh.pf) {
                std::cout << "ERROR: Jacobi iterations for solution of Vx not converging. Aborting" << std::endl;
            }
            MPI_Finalize();
            exit(0);
        }
    }
}


/**
 ********************************************************************************************************************************************
 * \brief   Function to solve the implicit equation for y-velocity
 *
 *          The implicit equation for \f$ u_y' \f$ of the implicit Crank-Nicholson method is solved using the Jacobi
 *          iterative method here.
 *
 *          The loop exits when the global maximum of the error in computed solution falls below the specified tolerance.
 *          If the solution doesn't converge even after an internally assigned maximum number for iterations, the solver
 *          aborts with an error message.
 *
 ********************************************************************************************************************************************
 */
void timestep::solveVy(vfield &V, plainvf &nseRHS, real beta) {
    int iterCount = 0;
    real locMax = 0.0;
    real gloMax = 0.0;

    while (true) {
#pragma omp parallel for num_threads(mesh.inputParams.nThreads) default(none) shared(V) shared(nseRHS) shared(beta)
        for (int iX = xSt; iX <= xEn; iX++) {
            for (int iY = ySt; iY <= yEn; iY++) {
                for (int iZ = zSt; iZ <= zEn; iZ++) {
                    iterTemp(iX, iY, iZ) = ((ihx2 * mesh.xix2(iX) * (V.Vy.F(iX+1, iY, iZ) + V.Vy.F(iX-1, iY, iZ)) +
                                             i2hx * mesh.xixx(iX) * (V.Vy.F(iX+1, iY, iZ) - V.Vy.F(iX-1, iY, iZ)) +
                                             ihy2 * mesh.ety2(iY) * (V.Vy.F(iX, iY+1, iZ) + V.Vy.F(iX, iY-1, iZ)) +
                                             i2hy * mesh.etyy(iY) * (V.Vy.F(iX, iY+1, iZ) - V.Vy.F(iX, iY-1, iZ)) +
                                             ihz2 * mesh.ztz2(iZ) * (V.Vy.F(iX, iY, iZ+1) + V.Vy.F(iX, iY, iZ-1)) +
                                             i2hz * mesh.ztzz(iZ) * (V.Vy.F(iX, iY, iZ+1) - V.Vy.F(iX, iY, iZ-1))) *
                            dt * nu * beta + nseRHS.Vy(iX, iY, iZ)) /
               (1.0 + 2.0 * dt * nu * beta * (ihx2 * mesh.xix2(iX) + ihy2 * mesh.ety2(iY) + ihz2 * mesh.ztz2(iZ)));
                }
            }
        }

        V.Vy.F = iterTemp;

        V.imposeVyBC();

#pragma omp parallel for num_threads(mesh.inputParams.nThreads) default(none) shared(V) shared(beta)
        for (int iX = xSt; iX <= xEn; iX++) {
            for (int iY = ySt; iY <= yEn; iY++) {
                for (int iZ = zSt; iZ <= zEn; iZ++) {
                    iterTemp(iX, iY, iZ) = V.Vy.F(iX, iY, iZ) - beta * dt * nu * (
                              mesh.xix2(iX) * (V.Vy.F(iX+1, iY, iZ) - 2.0 * V.Vy.F(iX, iY, iZ) + V.Vy.F(iX-1, iY, iZ)) * ihx2 +
                              mesh.xixx(iX) * (V.Vy.F(iX+1, iY, iZ) - V.Vy.F(iX-1, iY, iZ)) * i2hx +
                              mesh.ety2(iY) * (V.Vy.F(iX, iY+1, iZ) - 2.0 * V.Vy.F(iX, iY, iZ) + V.Vy.F(iX, iY-1, iZ)) * ihy2 +
                              mesh.etyy(iY) * (V.Vy.F(iX, iY+1, iZ) - V.Vy.F(iX, iY-1, iZ)) * i2hy +
                              mesh.ztz2(iZ) * (V.Vy.F(iX, iY, iZ+1) - 2.0 * V.Vy.F(iX, iY, iZ) + V.Vy.F(iX, iY, iZ-1)) * ihz2 +
                              mesh.ztzz(iZ) * (V.Vy.F(iX, iY, iZ+1) - V.Vy.F(iX, iY, iZ-1)) * i2hz);
                }
            }
        }

        iterTemp(core) = abs(iterTemp(core) - nseRHS.Vy(core));

        locMax = blitz::max(iterTemp(core));
        MPI_Allreduce(&locMax, &gloMax, 1, MPI_FP_REAL, MPI_MAX, MPI_COMM_WORLD);

        if (gloMax < mesh.inputParams.cnTolerance) break;

        iterCount += 1;

        if (iterCount > maxIterations) {
            if (mesh.pf) {
                std::cout << "ERROR: Jacobi iterations for solution of Vy not converging. Aborting" << std::endl;
            }
            MPI_Finalize();
            exit(0);
        }
    }
}


/**
 ********************************************************************************************************************************************
 * \brief   Function to solve the implicit equation for z-velocity
 *
 *          The implicit equation for \f$ u_z' \f$ of the implicit Crank-Nicholson method is solved using the Jacobi
 *          iterative method here.
 *
 *          The loop exits when the global maximum of the error in computed solution falls below the specified tolerance.
 *          If the solution doesn't converge even after an internally assigned maximum number for iterations, the solver
 *          aborts with an error message.
 *
 ********************************************************************************************************************************************
 */
void timestep::solveVz(vfield &V, plainvf &nseRHS, real beta) {
    int iterCount = 0;
    real locMax = 0.0;
    real gloMax = 0.0;

    while (true) {
#ifdef PLANAR
        int iY = 0;
#pragma omp parallel for num_threads(mesh.inputParams.nThreads) default(none) shared(V) shared(nseRHS) shared(iY) shared(beta)
        for (int iX = xSt; iX <= xEn; iX++) {
            for (int iZ = zSt; iZ <= zEn; iZ++) {
                iterTemp(iX, iY, iZ) = ((ihx2 * mesh.xix2(iX) * (V.Vz.F(iX+1, iY, iZ) + V.Vz.F(iX-1, iY, iZ)) +
                                       i2hx * mesh.xixx(iX) * (V.Vz.F(iX+1, iY, iZ) - V.Vz.F(iX-1, iY, iZ)) +
                                       ihz2 * mesh.ztz2(iZ) * (V.Vz.F(iX, iY, iZ+1) + V.Vz.F(iX, iY, iZ-1)) +
                                       i2hz * mesh.ztzz(iZ) * (V.Vz.F(iX, iY, iZ+1) - V.Vz.F(iX, iY, iZ-1))) *
                        dt * nu * beta + nseRHS.Vz(iX, iY, iZ)) /
           (1.0 + 2.0 * dt * nu * beta * (ihx2 * mesh.xix2(iX) + ihz2 * mesh.ztz2(iZ)));
            }
        }
#else
#pragma omp parallel for num_threads(mesh.inputParams.nThreads) default(none) shared(V) shared(nseRHS) shared(beta)
        for (int iX = xSt; iX <= xEn; iX++) {
            for (int iY = ySt; iY <= yEn; iY++) {
                for (int iZ = zSt; iZ <= zEn; iZ++) {
                    iterTemp(iX, iY, iZ) = ((ihx2 * mesh.xix2(iX) * (V.Vz.F(iX+1, iY, iZ) + V.Vz.F(iX-1, iY, iZ)) +
                                             i2hx * mesh.xixx(iX) * (V.Vz.F(iX+1, iY, iZ) - V.Vz.F(iX-1, iY, iZ)) +
                                             ihy2 * mesh.ety2(iY) * (V.Vz.F(iX, iY+1, iZ) + V.Vz.F(iX, iY-1, iZ)) +
                                             i2hy * mesh.etyy(iY) * (V.Vz.F(iX, iY+1, iZ) - V.Vz.F(iX, iY-1, iZ)) +
                                             ihz2 * mesh.ztz2(iZ) * (V.Vz.F(iX, iY, iZ+1) + V.Vz.F(iX, iY, iZ-1)) +
                                             i2hz * mesh.ztzz(iZ) * (V.Vz.F(iX, iY, iZ+1) - V.Vz.F(iX, iY, iZ-1))) *
                            dt * nu * beta + nseRHS.Vz(iX, iY, iZ)) /
               (1.0 + 2.0 * dt * nu * beta * (ihx2 * mesh.xix2(iX) + ihy2 * mesh.ety2(iY) + ihz2 * mesh.ztz2(iZ)));
                }
            }
        }
#endif

        V.Vz.F = iterTemp;

        V.imposeVzBC();

#ifdef PLANAR
#pragma omp parallel for num_threads(mesh.inputParams.nThreads) default(none) shared(V) shared(iY) shared(beta)
        for (int iX = xSt; iX <= xEn; iX++) {
            for (int iZ = zSt; iZ <= zEn; iZ++) {
                iterTemp(iX, iY, iZ) = V.Vz.F(iX, iY, iZ) - beta * dt * nu * (
                          mesh.xix2(iX) * (V.Vz.F(iX+1, iY, iZ) - 2.0 * V.Vz.F(iX, iY, iZ) + V.Vz.F(iX-1, iY, iZ)) * ihx2 +
                          mesh.xixx(iX) * (V.Vz.F(iX+1, iY, iZ) - V.Vz.F(iX-1, iY, iZ)) * i2hx +
                          mesh.ztz2(iZ) * (V.Vz.F(iX, iY, iZ+1) - 2.0 * V.Vz.F(iX, iY, iZ) + V.Vz.F(iX, iY, iZ-1)) * ihz2 +
                          mesh.ztzz(iZ) * (V.Vz.F(iX, iY, iZ+1) - V.Vz.F(iX, iY, iZ-1)) * i2hz);
            }
        }
#else
#pragma omp parallel for num_threads(mesh.inputParams.nThreads) default(none) shared(V) shared(beta)
        for (int iX = xSt; iX <= xEn; iX++) {
            for (int iY = ySt; iY <= yEn; iY++) {
                for (int iZ = zSt; iZ <= zEn; iZ++) {
                    iterTemp(iX, iY, iZ) = V.Vz.F(iX, iY, iZ) - beta * dt * nu * (
                              mesh.xix2(iX) * (V.Vz.F(iX+1, iY, iZ) - 2.0 * V.Vz.F(iX, iY, iZ) + V.Vz.F(iX-1, iY, iZ)) * ihx2 +
                              mesh.xixx(iX) * (V.Vz.F(iX+1, iY, iZ) - V.Vz.F(iX-1, iY, iZ)) * i2hx +
                              mesh.ety2(iY) * (V.Vz.F(iX, iY+1, iZ) - 2.0 * V.Vz.F(iX, iY, iZ) + V.Vz.F(iX, iY-1, iZ)) * ihy2 +
                              mesh.etyy(iY) * (V.Vz.F(iX, iY+1, iZ) - V.Vz.F(iX, iY-1, iZ)) * i2hy +
                              mesh.ztz2(iZ) * (V.Vz.F(iX, iY, iZ+1) - 2.0 * V.Vz.F(iX, iY, iZ) + V.Vz.F(iX, iY, iZ-1)) * ihz2 +
                              mesh.ztzz(iZ) * (V.Vz.F(iX, iY, iZ+1) - V.Vz.F(iX, iY, iZ-1)) * i2hz);
                }
            }
        }
#endif

        iterTemp(core) = abs(iterTemp(core) - nseRHS.Vz(core));

        locMax = blitz::max(iterTemp(core));
        MPI_Allreduce(&locMax, &gloMax, 1, MPI_FP_REAL, MPI_MAX, MPI_COMM_WORLD);

        if (gloMax < mesh.inputParams.cnTolerance) break;

        iterCount += 1;

        if (iterCount > maxIterations) {
            if (mesh.pf) {
                std::cout << "ERROR: Jacobi iterations for solution of Vz not converging. Aborting" << std::endl;
            }
            MPI_Finalize();
            exit(0);
        }
    }
}


/**
 ********************************************************************************************************************************************
 * \brief   Function to solve the implicit equation for scalar field
 *
 *          The implicit equation for \f$ \theta' \f$ of the implicit Crank-Nicholson method is solved using the Jacobi
 *          iterative method here.
 *
 *          The loop exits when the global maximum of the error in computed solution falls below the specified tolerance.
 *          If the solution doesn't converge even after an internally assigned maximum number for iterations, the solver
 *          aborts with an error message.
 *
 ********************************************************************************************************************************************
 */
void timestep::solveT(sfield &T, plainsf &tmpRHS, real beta) {
    int iterCount = 0;
    real locMax = 0.0;
    real gloMax = 0.0;

    while (true) {
#ifdef PLANAR
        int iY = 0;
#pragma omp parallel for num_threads(mesh.inputParams.nThreads) default(none) shared(T) shared(tmpRHS) shared(iY) shared(beta)
        for (int iX = xSt; iX <= xEn; iX++) {
            for (int iZ = zSt; iZ <= zEn; iZ++) {
                iterTemp(iX, iY, iZ) = ((ihx2 * mesh.xix2(iX) * (T.F.F(iX+1, iY, iZ) + T.F.F(iX-1, iY, iZ)) +
                                      i2hx * mesh.xixx(iX) * (T.F.F(iX+1, iY, iZ) - T.F.F(iX-1, iY, iZ)) +
                                      ihz2 * mesh.ztz2(iZ) * (T.F.F(iX, iY, iZ+1) + T.F.F(iX, iY, iZ-1)) +
                                      i2hz * mesh.ztzz(iZ) * (T.F.F(iX, iY, iZ+1) - T.F.F(iX, iY, iZ-1))) *
                    dt * kappa * beta + tmpRHS.F(iX, iY, iZ)) /
       (1.0 + 2.0 * dt * kappa * beta * (ihx2 * mesh.xix2(iX) + ihz2 * mesh.ztz2(iZ)));
            }
        }
#else
#pragma omp parallel for num_threads(mesh.inputParams.nThreads) default(none) shared(T) shared(tmpRHS) shared(beta)
        for (int iX = xSt; iX <= xEn; iX++) {
            for (int iY = ySt; iY <= yEn; iY++) {
                for (int iZ = zSt; iZ <= zEn; iZ++) {
                    iterTemp(iX, iY, iZ) = ((ihx2 * mesh.xix2(iX) * (T.F.F(iX+1, iY, iZ) + T.F.F(iX-1, iY, iZ)) +
                                             i2hx * mesh.xixx(iX) * (T.F.F(iX+1, iY, iZ) - T.F.F(iX-1, iY, iZ)) +
                                             ihy2 * mesh.ety2(iY) * (T.F.F(iX, iY+1, iZ) + T.F.F(iX, iY-1, iZ)) +
                                             i2hy * mesh.etyy(iY) * (T.F.F(iX, iY+1, iZ) - T.F.F(iX, iY-1, iZ)) +
                                             ihz2 * mesh.ztz2(iZ) * (T.F.F(iX, iY, iZ+1) + T.F.F(iX, iY, iZ-1)) +
                                             i2hz * mesh.ztzz(iZ) * (T.F.F(iX, iY, iZ+1) - T.F.F(iX, iY, iZ-1))) *
                        dt * kappa * beta + tmpRHS.F(iX, iY, iZ)) /
           (1.0 + 2.0 * dt * kappa * beta * (ihx2 * mesh.xix2(iX) + ihy2 * mesh.ety2(iY) + ihz2 * mesh.ztz2(iZ)));
                }
            }
        }
#endif

        T.F.F = iterTemp;

        T.imposeBCs();

#ifdef PLANAR
#pragma omp parallel for num_threads(mesh.inputParams.nThreads) default(none) shared(T) shared(iY) shared(beta)
        for (int iX = xSt; iX <= xEn; iX++) {
            for (int iZ = zSt; iZ <= zEn; iZ++) {
                iterTemp(iX, iY, iZ) = T.F.F(iX, iY, iZ) - beta * dt * kappa * (
                       mesh.xix2(iX) * (T.F.F(iX+1, iY, iZ) - 2.0 * T.F.F(iX, iY, iZ) + T.F.F(iX-1, iY, iZ)) * ihx2 +
                       mesh.xixx(iX) * (T.F.F(iX+1, iY, iZ) - T.F.F(iX-1, iY, iZ)) * i2hx +
                       mesh.ztz2(iZ) * (T.F.F(iX, iY, iZ+1) - 2.0 * T.F.F(iX, iY, iZ) + T.F.F(iX, iY, iZ-1)) * ihz2 +
                       mesh.ztzz(iZ) * (T.F.F(iX, iY, iZ+1) - T.F.F(iX, iY, iZ-1)) * i2hz);
            }
        }
#else
#pragma omp parallel for num_threads(mesh.inputParams.nThreads) default(none) shared(T) shared(beta)
        for (int iX = xSt; iX <= xEn; iX++) {
            for (int iY = ySt; iY <= yEn; iY++) {
                for (int iZ = zSt; iZ <= zEn; iZ++) {
                    iterTemp(iX, iY, iZ) = T.F.F(iX, iY, iZ) - beta * dt * kappa * (
                           mesh.xix2(iX) * (T.F.F(iX+1, iY, iZ) - 2.0 * T.F.F(iX, iY, iZ) + T.F.F(iX-1, iY, iZ)) * ihx2 +
                           mesh.xixx(iX) * (T.F.F(iX+1, iY, iZ) - T.F.F(iX-1, iY, iZ)) * i2hx +
                           mesh.ety2(iY) * (T.F.F(iX, iY+1, iZ) - 2.0 * T.F.F(iX, iY, iZ) + T.F.F(iX, iY-1, iZ)) * ihy2 +
                           mesh.etyy(iY) * (T.F.F(iX, iY+1, iZ) - T.F.F(iX, iY-1, iZ)) * i2hy +
                           mesh.ztz2(iZ) * (T.F.F(iX, iY, iZ+1) - 2.0 * T.F.F(iX, iY, iZ) + T.F.F(iX, iY, iZ-1)) * ihz2 +
                           mesh.ztzz(iZ) * (T.F.F(iX, iY, iZ+1) - T.F.F(iX, iY, iZ-1)) * i2hz);
                }
            }
        }
#endif

        iterTemp(core) = abs(iterTemp(core) - tmpRHS.F(core));

        locMax = blitz::max(iterTemp(core));
        MPI_Allreduce(&locMax, &gloMax, 1, MPI_FP_REAL, MPI_MAX, MPI_COMM_WORLD);

        if (gloMax < mesh.inputParams.cnTolerance) break;

        iterCount += 1;

        if (iterCount > maxIterations) {
            if (mesh.pf) {
                std::cout << "ERROR: Jacobi iterations for solution of T not converging. Aborting" << std::endl;
            }
            MPI_Finalize();
            exit(0);
        }
    }
}
