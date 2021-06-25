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
/*! \file lsRK3_d2.cc
 *
 *  \brief Definitions for functions of class timestep
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
 * \brief   Constructor of the timestep class
 *
 *          The empty constructor merely initializes the local reference to the global mesh variable.
 *          Also, the maximum allowable number of iterations for the Jacobi iterative solver being used to solve for the
 *          velocities implicitly is set as \f$ N_{max} = N_x \times N_y \times N_z \f$, where \f$N_x\f$, \f$N_y\f$ and \f$N_z\f$
 *          are the number of grid points in the collocated grid at the local sub-domains along x, y and z directions
 *          respectively.
 *
 * \param   mesh is a const reference to the global data contained in the grid class.
 ********************************************************************************************************************************************
 */
lsRK3_d2::lsRK3_d2(const grid &mesh, const real &sTime, const real &dt, tseries &tsIO, vfield &V, sfield &P):
    timestep(mesh, sTime, dt, tsIO, V, P),
    mgSolver(mesh, mesh.inputParams)
{
    setCoefficients();

    // This upper limit on max iterations is an arbitrarily chosen function.
    // Using Nx x Ny x Nz as the upper limit may cause the run to freeze for very long time.
    // This can eat away a lot of core hours unnecessarily.
    // It remains to be seen if this upper limit is safe.
    maxIterations = int(std::pow(std::log(mesh.coreSize(0)*mesh.coreSize(1)*mesh.coreSize(2)), 3));

    // These coefficients are taken from references [2], [3] and [4] of the Journal references in README
    alphRK3 = 4.0/15.0, 1.0/15.0, 1.0/6.0;
    betaRK3 = 4.0/15.0, 1.0/15.0, 1.0/6.0;
    gammRK3 = 8.0/15.0, 5.0/12.0, 3.0/4.0;
    zetaRK3 = 0.0, -17.0/60.0, -5.0/12.0;
}


/**
 ********************************************************************************************************************************************
 * \brief   Function to advance the solution using Euler method and Implicit Crank-Nicholson method
 *
 *          The non-linear terms are advanced using explicit Euler method, while the duffusion terms are
 *          advanced by semi-implicit Crank-Nicholson method.
 *          This overloaded function advances velocity and pressure fields for hydrodynamics simulations.
 *
 ********************************************************************************************************************************************
 */
void lsRK3_d2::timeAdvance(vfield &V, sfield &P) {
    int rkLev;

    static plainvf nseRHS(mesh);
    static plainvf nltVel(mesh);

    static vfield Vp(mesh, "Vp");

    Vp = V;

    for (rkLev = 0; rkLev < 3; rkLev++) {
        nseRHS = 0.0;
        nltVel = 0.0;

        // Compute the diffusion term for current sub-step
        V.computeDiff(nseRHS);
        nseRHS *= nu;

        // Add the velocity forcing term
        V.vForcing->addForcing(nseRHS);

        // Subtract the pressure gradient term
        pressureGradient = 0.0;
        P.gradient(pressureGradient);
        nseRHS -= pressureGradient;

        // Multiply all the collected linear terms of NSE with alpha coefficient
        nseRHS *= alphRK3(rkLev);

        // Compute the non-linear term for current sub-step
        V.computeNLin(V, nltVel);
        nltVel *= gammRK3(rkLev);

        // Add non-linear term to RHS
        nseRHS += nltVel;

        // Compute non-linear term of previous sub-step
        if (rkLev > 0) {
            nltVel = 0.0;

            Vp.computeNLin(Vp, nltVel);
            nltVel *= zetaRK3(rkLev);
            nseRHS += nltVel;

            Vp = V;
        }

        // Multiply the entire RHS with dt and add the velocity of previous time-step
        nseRHS *= dt;
        nseRHS += V;

        // Synchronize the RHS term across all processors by updating its sub-domain pads
        nseRHS.syncData();

        // Using the RHS term computed, compute the guessed velocity of CN method iteratively (and store it in V)
        solveVx(V, nseRHS, betaRK3(rkLev));
        solveVz(V, nseRHS, betaRK3(rkLev));

        // Calculate the rhs for the poisson solver (mgRHS) using the divergence of guessed velocity in V
        V.divergence(mgRHS);
        mgRHS *= 1.0/dt;

        // Using the calculated mgRHS, evaluate pressure correction (Pp) using multi-grid method
        mgSolver.mgSolve(Pp, mgRHS);

        // Synchronise the pressure correction term across processors
        Pp.syncData();

        // Add the pressure correction term to the pressure field of previous time-step, P
        P += Pp;

        // Finally get the velocity field at end of time-step by subtracting the gradient of pressure correction from V
        Pp.gradient(pressureGradient);
        pressureGradient *= dt;
        V -= pressureGradient;

        // Impose boundary conditions on the updated velocity field, V
        V.imposeBCs();

        // Impose boundary conditions on the updated pressure field, P
        P.imposeBCs();
    }
}


/**
 ********************************************************************************************************************************************
 * \brief   Function to advance the solution using Euler method and Implicit Crank-Nicholson method
 *
 *          The non-linear terms are advanced using explicit Euler method, while the duffusion terms are
 *          advanced by semi-implicit Crank-Nicholson method.
 *          This overloaded function advances velocity, temperature and pressure fields for scalar simulations.
 *
 ********************************************************************************************************************************************
 */
void lsRK3_d2::timeAdvance(vfield &V, sfield &P, sfield &T) {
    int rkLev;

    static plainvf nseRHS(mesh);
    static plainvf nltVel(mesh);

    static plainsf tmpRHS(mesh);
    static plainsf nltTmp(mesh);

    static vfield Vp(mesh, "Vp");
    static sfield Tp(mesh, "Tp");

    Vp = V;
    Tp = T;

    for (rkLev = 0; rkLev < 3; rkLev++) {
        nseRHS = 0.0;
        tmpRHS = 0.0;
        nltTmp = 0.0;
        nltVel = 0.0;

        // Compute the diffusion term of momentum equation for current sub-step
        V.computeDiff(nseRHS);
        nseRHS *= nu;

        // Compute the diffusion term of scalar equation for current sub-step
        T.computeDiff(tmpRHS);
        tmpRHS *= kappa;

        // Add the velocity forcing term
        V.vForcing->addForcing(nseRHS);

        // Add the scalar forcing term
        T.tForcing->addForcing(tmpRHS);

        // Subtract the pressure gradient term from momentum equation
        pressureGradient = 0.0;
        P.gradient(pressureGradient);
        nseRHS -= pressureGradient;

        // Multiply all the collected linear terms with alpha coefficient
        nseRHS *= alphRK3(rkLev);
        tmpRHS *= alphRK3(rkLev);

        // Compute the non-linear term of momentum equation for current sub-step
        V.computeNLin(V, nltVel);
        nltVel *= gammRK3(rkLev);

        // Compute the non-linear term of scalar equation for current sub-step
        T.computeNLin(V, nltTmp);
        nltTmp *= gammRK3(rkLev);

        // Add non-linear terms
        nseRHS += nltVel;
        tmpRHS += nltTmp;

        // Compute non-linear terms of previous sub-step fields
        if (rkLev > 0) {
            nltVel = 0.0;
            nltTmp = 0.0;

            Vp.computeNLin(Vp, nltVel);
            Tp.computeNLin(Vp, nltTmp);

            nltVel *= zetaRK3(rkLev);
            nltTmp *= zetaRK3(rkLev);

            nseRHS += nltVel;
            tmpRHS += nltTmp;

            Vp = V;
            Tp = T;
        }

        // Multiply the entire RHS with dt and add the velocity of previous time-step
        nseRHS *= dt;
        nseRHS += V;

        // Multiply the entire RHS with dt and add the temperature of previous time-step to advance by explicit Euler method
        tmpRHS *= dt;
        tmpRHS += T;

        // Synchronize both the RHS terms across all processors by updating their sub-domain pads
        nseRHS.syncData();
        tmpRHS.syncData();

        // Using the RHS term computed, compute the guessed velocity of CN method iteratively (and store it in V)
        solveVx(V, nseRHS, betaRK3(rkLev));
        solveVz(V, nseRHS, betaRK3(rkLev));

        // Using the RHS term computed, compute the temperature at next time-step iteratively (and store it in T)
        solveT(T, tmpRHS, betaRK3(rkLev));

        // Calculate the rhs for the poisson solver (mgRHS) using the divergence of guessed velocity in V
        V.divergence(mgRHS);
        mgRHS *= 1.0/dt;

        // Using the calculated mgRHS, evaluate pressure correction (Pp) using multi-grid method
        mgSolver.mgSolve(Pp, mgRHS);

        // Synchronise the pressure correction term across processors
        Pp.syncData();

        // Add the pressure correction term to the pressure field of previous time-step, P
        P += Pp;

        // Finally get the velocity field at end of time-step by subtracting the gradient of pressure correction from V
        Pp.gradient(pressureGradient);
        pressureGradient *= dt;
        V -= pressureGradient;

        // Impose boundary conditions on the updated velocity field, V
        V.imposeBCs();

        // Impose boundary conditions on the updated pressure field, P
        P.imposeBCs();

        // Impose boundary conditions on the updated temperature field, T
        T.imposeBCs();
    }
}


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
void lsRK3_d2::solveVx(vfield &V, plainvf &nseRHS, real beta) {
    int iterCount = 0;
    real locMax = 0.0;
    real gloMax = 0.0;
    static blitz::Array<real, 3> tempVx(V.Vx.F.lbound(), V.Vx.F.shape());

    while (true) {
        int iY = 0;
        for (int iX = xSt; iX <= xEn; iX++) {
            for (int iZ = zSt; iZ <= zEn; iZ++) {
                tempVx(iX, iY, iZ) = ((ihx2 * mesh.xix2(iX) * (V.Vx.F(iX+1, iY, iZ) + V.Vx.F(iX-1, iY, iZ)) +
                                       i2hx * mesh.xixx(iX) * (V.Vx.F(iX+1, iY, iZ) - V.Vx.F(iX-1, iY, iZ)) +
                                       ihz2 * mesh.ztz2(iZ) * (V.Vx.F(iX, iY, iZ+1) + V.Vx.F(iX, iY, iZ-1)) +
                                       i2hz * mesh.ztzz(iZ) * (V.Vx.F(iX, iY, iZ+1) - V.Vx.F(iX, iY, iZ-1))) *
                        dt * nu * beta + nseRHS.Vx(iX, iY, iZ)) /
           (1.0 + 2.0 * dt * nu * beta * (ihx2 * mesh.xix2(iX) + ihz2 * mesh.ztz2(iZ)));
            }
        }

        V.Vx.F = tempVx;

        V.imposeVxBC();

        for (int iX = xSt; iX <= xEn; iX++) {
            for (int iZ = zSt; iZ <= zEn; iZ++) {
                tempVx(iX, iY, iZ) = V.Vx.F(iX, iY, iZ) - beta * dt * nu * (
                          mesh.xix2(iX) * (V.Vx.F(iX+1, iY, iZ) - 2.0 * V.Vx.F(iX, iY, iZ) + V.Vx.F(iX-1, iY, iZ)) * ihx2 +
                          mesh.xixx(iX) * (V.Vx.F(iX+1, iY, iZ) - V.Vx.F(iX-1, iY, iZ)) * i2hx +
                          mesh.ztz2(iZ) * (V.Vx.F(iX, iY, iZ+1) - 2.0 * V.Vx.F(iX, iY, iZ) + V.Vx.F(iX, iY, iZ-1)) * ihz2 +
                          mesh.ztzz(iZ) * (V.Vx.F(iX, iY, iZ+1) - V.Vx.F(iX, iY, iZ-1)) * i2hz);
            }
        }

        tempVx(core) = abs(tempVx(core) - nseRHS.Vx(core));

        locMax = blitz::max(tempVx(core));
        MPI_Allreduce(&locMax, &gloMax, 1, MPI_FP_REAL, MPI_MAX, MPI_COMM_WORLD);

        if (gloMax < mesh.inputParams.cnTolerance) break;

        iterCount += 1;

        if (iterCount > maxIterations) {
            if (mesh.rankData.rank == 0) {
                std::cout << "ERROR: Jacobi iterations for solution of Vx not converging. Aborting" << std::endl;
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
void lsRK3_d2::solveVz(vfield &V, plainvf &nseRHS, real beta) {
    int iterCount = 0;
    real locMax = 0.0;
    real gloMax = 0.0;
    static blitz::Array<real, 3> tempVz(V.Vz.F.lbound(), V.Vz.F.shape());

    while (true) {
        int iY = 0;
        for (int iX = xSt; iX <= xEn; iX++) {
            for (int iZ = zSt; iZ <= zEn; iZ++) {
                tempVz(iX, iY, iZ) = ((ihx2 * mesh.xix2(iX) * (V.Vz.F(iX+1, iY, iZ) + V.Vz.F(iX-1, iY, iZ)) +
                                       i2hx * mesh.xixx(iX) * (V.Vz.F(iX+1, iY, iZ) - V.Vz.F(iX-1, iY, iZ)) +
                                       ihz2 * mesh.ztz2(iZ) * (V.Vz.F(iX, iY, iZ+1) + V.Vz.F(iX, iY, iZ-1)) +
                                       i2hz * mesh.ztzz(iZ) * (V.Vz.F(iX, iY, iZ+1) - V.Vz.F(iX, iY, iZ-1))) *
                        dt * nu * beta + nseRHS.Vz(iX, iY, iZ)) /
           (1.0 + 2.0 * dt * nu * beta * (ihx2 * mesh.xix2(iX) + ihz2 * mesh.ztz2(iZ)));
            }
        }

        V.Vz.F = tempVz;

        V.imposeVzBC();

        for (int iX = xSt; iX <= xEn; iX++) {
            for (int iZ = zSt; iZ <= zEn; iZ++) {
                tempVz(iX, iY, iZ) = V.Vz.F(iX, iY, iZ) - beta * dt * nu * (
                          mesh.xix2(iX) * (V.Vz.F(iX+1, iY, iZ) - 2.0 * V.Vz.F(iX, iY, iZ) + V.Vz.F(iX-1, iY, iZ)) * ihx2 +
                          mesh.xixx(iX) * (V.Vz.F(iX+1, iY, iZ) - V.Vz.F(iX-1, iY, iZ)) * i2hx +
                          mesh.ztz2(iZ) * (V.Vz.F(iX, iY, iZ+1) - 2.0 * V.Vz.F(iX, iY, iZ) + V.Vz.F(iX, iY, iZ-1)) * ihz2 +
                          mesh.ztzz(iZ) * (V.Vz.F(iX, iY, iZ+1) - V.Vz.F(iX, iY, iZ-1)) * i2hz);
            }
        }

        tempVz(core) = abs(tempVz(core) - nseRHS.Vz(core));

        locMax = blitz::max(tempVz(core));
        MPI_Allreduce(&locMax, &gloMax, 1, MPI_FP_REAL, MPI_MAX, MPI_COMM_WORLD);

        if (gloMax < mesh.inputParams.cnTolerance) break;

        iterCount += 1;

        if (iterCount > maxIterations) {
            if (mesh.rankData.rank == 0) {
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
void lsRK3_d2::solveT(sfield &T, plainsf &tmpRHS, real beta) {
    int iterCount = 0;
    real locMax = 0.0;
    real gloMax = 0.0;
    static blitz::Array<real, 3> tempT(T.F.F.lbound(), T.F.F.shape());

    while (true) {
        int iY = 0;
        for (int iX = xSt; iX <= xEn; iX++) {
            for (int iZ = zSt; iZ <= zEn; iZ++) {
                tempT(iX, iY, iZ) = ((ihx2 * mesh.xix2(iX) * (T.F.F(iX+1, iY, iZ) + T.F.F(iX-1, iY, iZ)) +
                                      i2hx * mesh.xixx(iX) * (T.F.F(iX+1, iY, iZ) - T.F.F(iX-1, iY, iZ)) +
                                      ihz2 * mesh.ztz2(iZ) * (T.F.F(iX, iY, iZ+1) + T.F.F(iX, iY, iZ-1)) +
                                      i2hz * mesh.ztzz(iZ) * (T.F.F(iX, iY, iZ+1) - T.F.F(iX, iY, iZ-1))) *
                    dt * kappa * beta + tmpRHS.F(iX, iY, iZ)) /
       (1.0 + 2.0 * dt * kappa * beta * (ihx2 * mesh.xix2(iX) + ihz2 * mesh.ztz2(iZ)));
            }
        }

        T.F.F = tempT;

        T.imposeBCs();

        for (int iX = xSt; iX <= xEn; iX++) {
            for (int iZ = zSt; iZ <= zEn; iZ++) {
                tempT(iX, iY, iZ) = T.F.F(iX, iY, iZ) - beta * dt * kappa * (
                       mesh.xix2(iX) * (T.F.F(iX+1, iY, iZ) - 2.0 * T.F.F(iX, iY, iZ) + T.F.F(iX-1, iY, iZ)) * ihx2 +
                       mesh.xixx(iX) * (T.F.F(iX+1, iY, iZ) - T.F.F(iX-1, iY, iZ)) * i2hx +
                       mesh.ztz2(iZ) * (T.F.F(iX, iY, iZ+1) - 2.0 * T.F.F(iX, iY, iZ) + T.F.F(iX, iY, iZ-1)) * ihz2 +
                       mesh.ztzz(iZ) * (T.F.F(iX, iY, iZ+1) - T.F.F(iX, iY, iZ-1)) * i2hz);
            }
        }

        tempT(core) = abs(tempT(core) - tmpRHS.F(core));

        locMax = blitz::max(tempT(core));
        MPI_Allreduce(&locMax, &gloMax, 1, MPI_FP_REAL, MPI_MAX, MPI_COMM_WORLD);

        if (gloMax < mesh.inputParams.cnTolerance) break;

        iterCount += 1;

        if (iterCount > maxIterations) {
            if (mesh.rankData.rank == 0) {
                std::cout << "ERROR: Jacobi iterations for solution of T not converging. Aborting" << std::endl;
            }
            MPI_Finalize();
            exit(0);
        }
    }
}


/**
 ********************************************************************************************************************************************
 * \brief   Function to set the coefficients used for solving the implicit equations of U, V and W
 *
 *          The function assigns values to the variables \ref hx, \ref hy, etc.
 *          These coefficients are repeatedly used at many places in the iterative solver for implicit calculation of velocities.
 ********************************************************************************************************************************************
 */
void lsRK3_d2::setCoefficients() {
    real hx2 = pow(mesh.dXi, 2.0);
    real hz2 = pow(mesh.dZt, 2.0);

    i2hx = 0.5/mesh.dXi;
    i2hz = 0.5/mesh.dZt;

    ihx2 = 1.0/hx2;
    ihz2 = 1.0/hz2;
};