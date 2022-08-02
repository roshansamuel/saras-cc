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
/*! \file lsRK3_d3.cc
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
lsRK3_d3::lsRK3_d3(const grid &mesh, const real &sTime, const real &dt, tseries &tsIO, vfield &V, sfield &P):
    timestep(mesh, sTime, dt, tsIO, V, P),
    mgSolver(mesh, mesh.inputParams)
{
    // These coefficients are taken from references [2], [3] and [4] of the Journal references in README
    alphRK3 = 4.0/15.0, 1.0/15.0, 1.0/6.0;
    betaRK3 = 4.0/15.0, 1.0/15.0, 1.0/6.0;
    gammRK3 = 8.0/15.0, 5.0/12.0, 3.0/4.0;
    zetaRK3 = 0.0, -17.0/60.0, -5.0/12.0;

    /** The paper by Spalart ([2] in Journal references of README) uses different coefficients for alpha and beta
    alphRK3 = 29.0/96.0, -3.0/40.0, 1.0/6.0;
    betaRK3 = 37.0/160.0, 5.0/24.0, 1.0/6.0;
    **/

    // If LES switch is enabled, initialize LES model
    if (mesh.inputParams.lesModel) {
        if (mesh.pf) {
            std::cout << "LES Switch is ON. Using stretched spiral vortex LES Model\n" << std::endl;
        }

        sgsLES = new spiral(mesh, nu);
    }
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
void lsRK3_d3::timeAdvance(vfield &V, sfield &P) {
    int rkLev;

    static plainvf nseRHS(mesh);
    static plainvf tempVF(mesh);

    for (rkLev = 0; rkLev < 3; rkLev++) {
        nseRHS = 0.0;

        // Add the contribution from previous sub-step non-linear term
        if (rkLev > 0) nseRHS = nseRHS.multAdd(tempVF, zetaRK3(rkLev));

        tempVF = 0.0;

        // Compute the diffusion term (linear term) for current sub-step
        V.computeDiff(tempVF);
        nseRHS = nseRHS.multAdd(tempVF, alphRK3(rkLev)*nu);

        tempVF = 0.0;

        // Add the velocity forcing term
        V.vForcing->addForcing(tempVF);

        // Subtract the pressure gradient term
        pressureGradient = 0.0;
        P.gradient(pressureGradient);
        tempVF -= pressureGradient;

        // Add the forcing and pressure gradient terms to the RHS with weights
        nseRHS = nseRHS.multAdd(tempVF, alphRK3(rkLev) + betaRK3(rkLev));

        tempVF = 0.0;

        // Compute the non-linear term for current sub-step
        V.computeNLin(V, tempVF);

        // Add sub-grid stress contribution from LES Model, if enabled
        if (mesh.inputParams.lesModel and solTime > 5*mesh.inputParams.tStp) {
            sgsLES->computeSG(tempVF, V);
            tsWriter.subgridEnergy = sgsLES->totalSGKE;
            tsWriter.sgDissipation = sgsLES->totalDisp;
            tsWriter.nuTurbulent = sgsLES->totalNuSG;
        }

        // Add non-linear term to RHS
        nseRHS = nseRHS.multAdd(tempVF, gammRK3(rkLev));

        // Multiply the entire RHS with dt and add the velocity of previous time-step
        nseRHS *= dt;
        nseRHS += V;

        // Synchronize the RHS term across all processors by updating its sub-domain pads
        nseRHS.syncFaces();

        // Using the RHS term computed, compute the guessed velocity of CN method iteratively (and store it in V)
        solveVx(V, nseRHS, betaRK3(rkLev));
        solveVy(V, nseRHS, betaRK3(rkLev));
        solveVz(V, nseRHS, betaRK3(rkLev));

        // Calculate the rhs for the poisson solver (mgRHS) using the divergence of guessed velocity in V
        V.divergence(mgRHS);
        mgRHS *= 1.0/((alphRK3(rkLev) + betaRK3(rkLev))*dt);

        // Using the calculated mgRHS, evaluate pressure correction (Pp) using multi-grid method
        mgSolver.mgSolve(Pp, mgRHS);

        // Synchronise the pressure correction term across processors
        Pp.syncFaces();

        // Add the pressure correction term to the pressure field of previous time-step, P
        P += Pp;

        // Finally get the velocity field at end of time-step by subtracting the gradient of pressure correction from V
        Pp.gradient(pressureGradient);
        pressureGradient *= (alphRK3(rkLev) + betaRK3(rkLev))*dt;
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
void lsRK3_d3::timeAdvance(vfield &V, sfield &P, sfield &T) {
    int rkLev;

    static plainvf nseRHS(mesh);
    static plainvf tempVF(mesh);

    static plainsf tmpRHS(mesh);
    static plainsf tempSF(mesh);

    for (rkLev = 0; rkLev < 3; rkLev++) {
        nseRHS = 0.0;
        tmpRHS = 0.0;

        // Add the contribution from previous sub-step non-linear terms
        if (rkLev > 0) {
            nseRHS = nseRHS.multAdd(tempVF, zetaRK3(rkLev));
            tmpRHS = tmpRHS.multAdd(tempSF, zetaRK3(rkLev));
        }

        tempVF = 0.0;
        tempSF = 0.0;

        // Compute the diffusion terms (linear terms) for current sub-step
        V.computeDiff(tempVF);
        T.computeDiff(tempSF);

        nseRHS = nseRHS.multAdd(tempVF, alphRK3(rkLev)*nu);
        tmpRHS = tmpRHS.multAdd(tempSF, alphRK3(rkLev)*kappa);

        tempVF = 0.0;
        tempSF = 0.0;

        // Add the forcing terms
        V.vForcing->addForcing(tempVF);
        T.tForcing->addForcing(tempSF);

        // Subtract the pressure gradient term
        pressureGradient = 0.0;
        P.gradient(pressureGradient);
        tempVF -= pressureGradient;

        // Add the forcing and pressure gradient terms to the RHS with weights
        nseRHS = nseRHS.multAdd(tempVF, alphRK3(rkLev) + betaRK3(rkLev));
        tmpRHS = tmpRHS.multAdd(tempSF, alphRK3(rkLev) + betaRK3(rkLev));

        tempVF = 0.0;
        tempSF = 0.0;

        // Compute the non-linear terms for current sub-step
        V.computeNLin(V, tempVF);
        T.computeNLin(V, tempSF);

        // Add sub-grid stress contribution from LES Model to the non-linear term, if enabled
        if (mesh.inputParams.lesModel and solTime > 5*mesh.inputParams.tStp) {
            if (mesh.inputParams.lesModel == 1)
                sgsLES->computeSG(tempVF, V);
            else if (mesh.inputParams.lesModel == 2)
                sgsLES->computeSG(tempVF, tempSF, V, T);

            tsWriter.subgridEnergy = sgsLES->totalSGKE;
            tsWriter.sgDissipation = sgsLES->totalDisp;
            tsWriter.nuTurbulent = sgsLES->totalNuSG;
        }

        // Add non-linear terms
        nseRHS = nseRHS.multAdd(tempVF, gammRK3(rkLev));
        tmpRHS = tmpRHS.multAdd(tempSF, gammRK3(rkLev));

        // Multiply the entire RHS with dt and add the velocity of previous time-step
        nseRHS *= dt;
        nseRHS += V;

        // Multiply the entire RHS with dt and add the temperature of previous time-step to advance by explicit Euler method
        tmpRHS *= dt;
        tmpRHS += T;

        // Synchronize both the RHS terms across all processors by updating their sub-domain pads
        nseRHS.syncFaces();
        tmpRHS.syncFaces();

        // Using the RHS term computed, compute the guessed velocity of CN method iteratively (and store it in V)
        solveVx(V, nseRHS, betaRK3(rkLev));
        solveVy(V, nseRHS, betaRK3(rkLev));
        solveVz(V, nseRHS, betaRK3(rkLev));

        // Using the RHS term computed, compute the temperature at next time-step iteratively (and store it in T)
        solveT(T, tmpRHS, betaRK3(rkLev));

        // Calculate the rhs for the poisson solver (mgRHS) using the divergence of guessed velocity in V
        V.divergence(mgRHS);
        mgRHS *= 1.0/((alphRK3(rkLev) + betaRK3(rkLev))*dt);

        // Using the calculated mgRHS, evaluate pressure correction (Pp) using multi-grid method
        mgSolver.mgSolve(Pp, mgRHS);

        // Synchronise the pressure correction term across processors
        Pp.syncFaces();

        // Add the pressure correction term to the pressure field of previous time-step, P
        P += Pp;

        // Finally get the velocity field at end of time-step by subtracting the gradient of pressure correction from V
        Pp.gradient(pressureGradient);
        pressureGradient *= (alphRK3(rkLev) + betaRK3(rkLev))*dt;
        V -= pressureGradient;

        // Impose boundary conditions on the updated velocity field, V
        V.imposeBCs();

        // Impose boundary conditions on the updated pressure field, P
        P.imposeBCs();

        // Impose boundary conditions on the updated temperature field, T
        T.imposeBCs();
    }
}

