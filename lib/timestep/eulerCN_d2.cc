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

