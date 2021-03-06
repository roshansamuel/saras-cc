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
/*! \file eulerCN_d2.cc
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
eulerCN_d2::eulerCN_d2(const grid &mesh, const real &sTime, const real &dt, tseries &tsIO, vfield &V, sfield &P):
    timestep(mesh, sTime, dt, tsIO, V, P),
    mgSolver(mesh, mesh.inputParams)
{
    setCoefficients();

    // This upper limit on max iterations is an arbitrarily chosen function.
    // Using Nx x Ny x Nz as the upper limit may cause the run to freeze for very long time.
    // This can eat away a lot of core hours unnecessarily.
    // It remains to be seen if this upper limit is safe.
    maxIterations = int(std::pow(std::log(mesh.coreSize(0)*mesh.coreSize(1)*mesh.coreSize(2)), 3));

    // These coefficients are default values for Crank-Nicholson
    // Sum of alpha and beta should be 1.0
    alphCN2 = 1.0/2.0;
    betaCN2 = 1.0/2.0;
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
void eulerCN_d2::timeAdvance(vfield &V, sfield &P) {
    static plainvf nseRHS(mesh);

    nseRHS = 0.0;

    // Compute the diffusion term of momentum equation
    V.computeDiff(nseRHS);
    // Split the diffusion term and multiply by diffusion coefficient
    nseRHS *= nu*alphCN2;

    // Compute the non-linear term and subtract it from the RHS
    V.computeNLin(V, nseRHS);

    // Add the velocity forcing term
    V.vForcing->addForcing(nseRHS);

    // Subtract the pressure gradient term
    pressureGradient = 0.0;
    P.gradient(pressureGradient);
    nseRHS -= pressureGradient;

    // Multiply the entire RHS with dt and add the velocity of previous time-step to advance by explicit Euler method
    nseRHS *= dt;
    nseRHS += V;

    // Synchronize the RHS term across all processors by updating its sub-domain pads
    nseRHS.syncFaces();

    // Using the RHS term computed, compute the guessed velocity of CN method iteratively (and store it in V)
    solveVx(V, nseRHS, betaCN2);
    solveVz(V, nseRHS, betaCN2);

    // Calculate the rhs for the poisson solver (mgRHS) using the divergence of guessed velocity in V
    V.divergence(mgRHS);
    mgRHS *= 1.0/dt;

    // IF THE POISSON SOLVER IS BEING TESTED, THE RHS IS SET TO ONE.
    // THIS IS FOR TESTING ONLY AND A SINGLE TIME ADVANCE IS PERFORMED IN THIS TEST
#ifdef TEST_POISSON
    mgRHS.F = 1.0;
#endif

    // Using the calculated mgRHS, evaluate pressure correction (Pp) using multi-grid method
    mgSolver.mgSolve(Pp, mgRHS);

    // Synchronise the pressure correction term across processors
    Pp.syncFaces();

    // IF THE POISSON SOLVER IS BEING TESTED, THE PRESSURE IS SET TO ZERO.
    // THIS WAY, AFTER THE SOLUTION OF MG SOLVER, Pp, IS DIRECTLY WRITTEN INTO P AND AVAILABLE FOR PLOTTING
    // THIS IS FOR TESTING ONLY AND A SINGLE TIME ADVANCE IS PERFORMED IN THIS TEST
#ifdef TEST_POISSON
    P.F = 0.0;
#endif

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
void eulerCN_d2::timeAdvance(vfield &V, sfield &P, sfield &T) {
    static plainvf nseRHS(mesh);
    static plainsf tmpRHS(mesh);

    // BELOW FLAG MAY BE TURNED OFF FOR DEBUGGING/DIGNOSTIC RUNS ONLY
    // IT IS USED TO TURN OFF COMPUTATION OF NON-LINEAR TERMS
    // CURRENTLY IT IS AVAILABLE ONLY FOR THE 2D SCALAR SOLVER
    bool nlinSwitch = true;

    nseRHS = 0.0;
    tmpRHS = 0.0;

    // Compute the diffusion term of momentum equation
    V.computeDiff(nseRHS);
    // Split the diffusion term and multiply by diffusion coefficient
    nseRHS *= nu*alphCN2;

    // Compute the diffusion term of scalar equation
    T.computeDiff(tmpRHS);
    // Split the diffusion term and multiply by diffusion coefficient
    tmpRHS *= kappa*alphCN2;

    // Compute the non-linear term and subtract it from the RHS of momentum equation
    V.computeNLin(V, nseRHS);

    if (nlinSwitch) {
        // Compute the non-linear term and subtract it from the RHS of scalar equation
        T.computeNLin(V, tmpRHS);

    } else {
        // EVEN WHEN NON-LINEAR TERM IS TURNED OFF, THE MEAN FLOW EFFECTS STILL REMAIN
        // HENCE THE CONTRIBUTION OF VELOCITY TO SCALAR EQUATION MUST BE ADDED
        // THIS CONTRIBUTION IS Uz FOR RBC AND SST, BUT Ux FOR VERTICAL CONVECTION
        if (mesh.inputParams.probType == 5 || mesh.inputParams.probType == 6) {
            tmpRHS.F += V.Vz.F;

        } else if (mesh.inputParams.probType == 7) {
            tmpRHS.F += V.Vx.F;
        }
    }

    // Add the velocity forcing term
    V.vForcing->addForcing(nseRHS);

    // Add the scalar forcing term
    T.tForcing->addForcing(tmpRHS);

    // Subtract the pressure gradient term from momentum equation
    pressureGradient = 0.0;
    P.gradient(pressureGradient);
    nseRHS -= pressureGradient;

    // Multiply the entire RHS with dt and add the velocity of previous time-step to advance by explicit Euler method
    nseRHS *= dt;
    nseRHS += V;

    // Multiply the entire RHS with dt and add the temperature of previous time-step to advance by explicit Euler method
    tmpRHS *= dt;
    tmpRHS += T;

    // Synchronize both the RHS terms across all processors by updating their sub-domain pads
    nseRHS.syncFaces();
    tmpRHS.syncFaces();

    // Using the RHS term computed, compute the guessed velocity of CN method iteratively (and store it in V)
    solveVx(V, nseRHS, betaCN2);
    solveVz(V, nseRHS, betaCN2);

    // Using the RHS term computed, compute the temperature at next time-step iteratively (and store it in T)
    solveT(T, tmpRHS, betaCN2);

    // Calculate the rhs for the poisson solver (mgRHS) using the divergence of guessed velocity in V
    V.divergence(mgRHS);
    mgRHS *= 1.0/dt;

    // Using the calculated mgRHS, evaluate pressure correction (Pp) using multi-grid method
    mgSolver.mgSolve(Pp, mgRHS);

    // Synchronise the pressure correction term across processors
    Pp.syncFaces();

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
void eulerCN_d2::solveVx(vfield &V, plainvf &nseRHS, real beta) {
    int iterCount = 0;
    real locMax = 0.0;
    real gloMax = 0.0;
    static blitz::Array<real, 3> tempVx(V.Vx.F.lbound(), V.Vx.F.shape());

    while (true) {
        int iY = 0;
#pragma omp parallel for num_threads(mesh.inputParams.nThreads) default(none) shared(V) shared(nseRHS) shared(tempVx) shared(iY) shared(beta)
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

#pragma omp parallel for num_threads(mesh.inputParams.nThreads) default(none) shared(V) shared(tempVx) shared(iY) shared(beta)
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
void eulerCN_d2::solveVz(vfield &V, plainvf &nseRHS, real beta) {
    int iterCount = 0;
    real locMax = 0.0;
    real gloMax = 0.0;
    static blitz::Array<real, 3> tempVz(V.Vz.F.lbound(), V.Vz.F.shape());

    while (true) {
        int iY = 0;
#pragma omp parallel for num_threads(mesh.inputParams.nThreads) default(none) shared(V) shared(nseRHS) shared(tempVz) shared(iY) shared(beta)
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

#pragma omp parallel for num_threads(mesh.inputParams.nThreads) default(none) shared(V) shared(tempVz) shared(iY) shared(beta)
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
void eulerCN_d2::solveT(sfield &T, plainsf &tmpRHS, real beta) {
    int iterCount = 0;
    real locMax = 0.0;
    real gloMax = 0.0;
    static blitz::Array<real, 3> tempT(T.F.F.lbound(), T.F.F.shape());

    while (true) {
        int iY = 0;
#pragma omp parallel for num_threads(mesh.inputParams.nThreads) default(none) shared(T) shared(tmpRHS) shared(tempT) shared(iY) shared(beta)
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

#pragma omp parallel for num_threads(mesh.inputParams.nThreads) default(none) shared(T) shared(tempT) shared(iY) shared(beta)
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
            if (mesh.pf) {
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
void eulerCN_d2::setCoefficients() {
    real hx2 = pow(mesh.dXi, 2.0);
    real hz2 = pow(mesh.dZt, 2.0);

    i2hx = 0.5/mesh.dXi;
    i2hz = 0.5/mesh.dZt;

    ihx2 = 1.0/hx2;
    ihz2 = 1.0/hz2;
};
