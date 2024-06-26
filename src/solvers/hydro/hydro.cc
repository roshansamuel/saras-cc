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
/*! \file hydro.cc
 *
 *  \brief Definitions of common functions for 2D and 3D versions of the solver class hydro - solves the basic Navier-Stokes equation.
 *  \sa hydro.h
 *  \author Roshan Samuel
 *  \date Nov 2019
 *  \copyright New BSD License
 *
 ********************************************************************************************************************************************
 */

#include "hydro.h"

/**
 ********************************************************************************************************************************************
 * \brief   Constructor of the base hydro class
 *
 *          The short base constructor of the hydro class merely assigns the const references to the grid and parser
 *          class instances being used in the solver.
 *
 * \param   mesh is a const reference to the global data contained in the grid class
 * \param   solParam is a const reference to the user-set parameters contained in the parser class
 ********************************************************************************************************************************************
 */
hydro::hydro(const grid &mesh, const parser &solParam, parallel &mpiParam):
            V(mesh, "V"),
            P(mesh, "P"),
            mesh(mesh),
            inputParams(solParam),
            rankData(mpiParam) {
    // For reasons I don't fully understand, an instance of plainvf
    // has to exist in the solver for the plainvf class to be
    // found by the compiler when linking the project.
    // The following object is not used anywhere.
    // However, if you remove it, the compiler will fail to link
    // plainvf class, and throw "undefined reference" error,
    // and that too in an entirely different part of the code.
    // This issue was found when using gcc 7.5
    plainvf why(mesh);

    if (inputParams.restartFlag) {
        int countVal;

        countVal = inputParams.fwInt/inputParams.tStp;
        if (countVal <= 2) {
            if (mesh.pf) std::cout << "ERROR: File-write interval is too small to estimate the next file-write time within floating point tolerance. Aborting" << std::endl;
            MPI_Finalize();
            exit(0);
        }

        countVal = inputParams.rsInt/inputParams.tStp;
        if (countVal <= 2) {
            if (mesh.pf) std::cout << "ERROR: Restart-write interval is too small to estimate the next restart-write time within floating point tolerance. Aborting" << std::endl;
            MPI_Finalize();
            exit(0);
        }
        
        if (inputParams.readProbes) {
            countVal = inputParams.prInt/inputParams.tStp;
            if (countVal <= 2) {
                if (mesh.pf) std::cout << "ERROR: Data-probe interval is too small to estimate the next data-probe time within floating point tolerance. Aborting" << std::endl;
                MPI_Finalize();
                exit(0);
            }
        }
    }
}


/**
 ********************************************************************************************************************************************
 * \brief   The core publicly accessible function of the \ref hydro class to solve the Navier-Stokes equations
 *
 *          The NSE are integrated in time from within this function by calling \ref hydro#timeAdvance in a loop.
 *          The function keeps track of the non-dimensional time with \ref time and number of iterations with \ref iterCount.
 *          Both these values are continuously incremented from within the loop, and finally, when \ref time has reached the
 *          user-set value in \ref parser#tMax "tMax", the time-integration loop is broken and the program exits.
 ********************************************************************************************************************************************
 */
void hydro::solvePDE() { };


/**
 ********************************************************************************************************************************************
 * \brief   Function to enable/disable periodic data transfer as per the problem
 *
 *          The function checks the xPer, yPer and zPer flags in the parser class
 *          and enables/disables MPI data transfer at boundaries accordingly
 *          By default, the MPI neighbours at boundaries are set for periodic data-transfer.
 *          This has to be disabled if the problem has non-periodic boundaries.
 *
 ********************************************************************************************************************************************
 */
void hydro::checkPeriodic() {
    // Disable periodic data transfer by setting neighbouring ranks of boundary sub-domains to NULL
    // Left and right walls
    if (not inputParams.xPer) {
        if (rankData.rank == 0) {
            std::cout << "Using non-periodic boundary conditions along X Direction" << std::endl;
            std::cout << std::endl;
        }

        if (rankData.xRank == 0) {
            rankData.faceRanks(0) = MPI_PROC_NULL;

            rankData.edgeRanks(0) = MPI_PROC_NULL;
            rankData.edgeRanks(1) = MPI_PROC_NULL;
            rankData.edgeRanks(8) = MPI_PROC_NULL;
            rankData.edgeRanks(10) = MPI_PROC_NULL;

            rankData.cornRanks(0) = MPI_PROC_NULL;
            rankData.cornRanks(1) = MPI_PROC_NULL;
            rankData.cornRanks(4) = MPI_PROC_NULL;
            rankData.cornRanks(5) = MPI_PROC_NULL;
        }

        if (rankData.xRank == rankData.npX-1) {
            rankData.faceRanks(1) = MPI_PROC_NULL;

            rankData.edgeRanks(2) = MPI_PROC_NULL;
            rankData.edgeRanks(3) = MPI_PROC_NULL;
            rankData.edgeRanks(9) = MPI_PROC_NULL;
            rankData.edgeRanks(11) = MPI_PROC_NULL;

            rankData.cornRanks(2) = MPI_PROC_NULL;
            rankData.cornRanks(3) = MPI_PROC_NULL;
            rankData.cornRanks(6) = MPI_PROC_NULL;
            rankData.cornRanks(7) = MPI_PROC_NULL;
        }
    }

    // Front and rear walls
#ifdef PLANAR
    // Front and rear walls are by default non-periodic for 2D simulations
    if (rankData.yRank == 0)                rankData.faceRanks(2) = MPI_PROC_NULL;
    if (rankData.yRank == rankData.npY-1)   rankData.faceRanks(3) = MPI_PROC_NULL;

#else
    if (not inputParams.yPer) {
        if (rankData.rank == 0) {
            std::cout << "Using non-periodic boundary conditions along Y Direction" << std::endl;
            std::cout << std::endl;
        }

        if (rankData.yRank == 0) {
            rankData.faceRanks(2) = MPI_PROC_NULL;

            rankData.edgeRanks(0) = MPI_PROC_NULL;
            rankData.edgeRanks(2) = MPI_PROC_NULL;
            rankData.edgeRanks(4) = MPI_PROC_NULL;
            rankData.edgeRanks(5) = MPI_PROC_NULL;

            rankData.cornRanks(0) = MPI_PROC_NULL;
            rankData.cornRanks(2) = MPI_PROC_NULL;
            rankData.cornRanks(4) = MPI_PROC_NULL;
            rankData.cornRanks(6) = MPI_PROC_NULL;
        }

        if (rankData.yRank == rankData.npY-1) {
            rankData.faceRanks(3) = MPI_PROC_NULL;

            rankData.edgeRanks(1) = MPI_PROC_NULL;
            rankData.edgeRanks(3) = MPI_PROC_NULL;
            rankData.edgeRanks(6) = MPI_PROC_NULL;
            rankData.edgeRanks(7) = MPI_PROC_NULL;

            rankData.cornRanks(1) = MPI_PROC_NULL;
            rankData.cornRanks(3) = MPI_PROC_NULL;
            rankData.cornRanks(5) = MPI_PROC_NULL;
            rankData.cornRanks(7) = MPI_PROC_NULL;
        }
    }
#endif

    // Top and bottom walls
    if (not inputParams.zPer) {
        if (rankData.rank == 0) {
            std::cout << "Using non-periodic boundary conditions along Z Direction" << std::endl;
            std::cout << std::endl;
        }

        if (rankData.zRank == 0) {
            rankData.faceRanks(4) = MPI_PROC_NULL;

            rankData.edgeRanks(4) = MPI_PROC_NULL;
            rankData.edgeRanks(6) = MPI_PROC_NULL;
            rankData.edgeRanks(8) = MPI_PROC_NULL;
            rankData.edgeRanks(9) = MPI_PROC_NULL;

            rankData.cornRanks(0) = MPI_PROC_NULL;
            rankData.cornRanks(1) = MPI_PROC_NULL;
            rankData.cornRanks(2) = MPI_PROC_NULL;
            rankData.cornRanks(3) = MPI_PROC_NULL;
        }

        if (rankData.zRank == rankData.npZ-1) {
            rankData.faceRanks(5) = MPI_PROC_NULL;

            rankData.edgeRanks(5) = MPI_PROC_NULL;
            rankData.edgeRanks(7) = MPI_PROC_NULL;
            rankData.edgeRanks(10) = MPI_PROC_NULL;
            rankData.edgeRanks(11) = MPI_PROC_NULL;

            rankData.cornRanks(4) = MPI_PROC_NULL;
            rankData.cornRanks(5) = MPI_PROC_NULL;
            rankData.cornRanks(6) = MPI_PROC_NULL;
            rankData.cornRanks(7) = MPI_PROC_NULL;
        }
    }
};


/**
 ********************************************************************************************************************************************
 * \brief   Function to initialize the forcing terms for velocity
 *
 *          The forcing terms for the velocity field are initialized here.
 *          Out of the different forcings available in the force class,
 *          the appropriate forcing is chosen according to the parameters set by the user.
 ********************************************************************************************************************************************
 */
void hydro::initVForcing() {
    switch (inputParams.forceType) {
        case 0:
            if (rankData.rank == 0) std::cout << "Running hydrodynamics simulation with zero velocity forcing" << std::endl << std::endl;
            V.vForcing = new zeroForcing(mesh, V);
            break;
        case 1:
            if (rankData.rank == 0) std::cout << "Running hydrodynamics simulation with random velocity forcing" << std::endl << std::endl;
            V.vForcing = new randomForcing(mesh, V);
            break;
        case 2:
            if (rankData.rank == 0) std::cout << "Running hydrodynamics simulation with rotation" << std::endl << std::endl;
            V.vForcing = new coriolisForce(mesh, V);
            break;
        case 5:
            if (rankData.rank == 0) std::cout << "Running hydrodynamics simulation with constant pressure gradient along X-direction" << std::endl << std::endl;
            V.vForcing = new constantPGrad(mesh, V);
            break;
        default:
            if (rankData.rank == 0) std::cout << "WARNING: Chosen velocity forcing is incompatible with hydrodynamics runs. Defaulting to zero forcing" << std::endl << std::endl;
            V.vForcing = new zeroForcing(mesh, V);
    }
}


/**
 ********************************************************************************************************************************************
 * \brief   Function to initialize the boundary conditions for velocity
 *
 *          The boundary conditions for all the 6 walls (4 in case of 2D simulations) are initialized here.
 *          Out of the different boundary conditions available in the boundary class,
 *          the appropriate BCs are chosen according to the type of problem being solved.
 ********************************************************************************************************************************************
 */
void hydro::initVBCs() {
#ifdef PLANAR
    const int nBCs = 4;
    const std::vector<int> wn = {0, 1, 4, 5};
#else
    const int nBCs = 6;
    const std::vector<int> wn = {0, 1, 2, 3, 4, 5};
#endif

    if (inputParams.useCustomBC) {
        auto uBCTypes = inputParams.getBCTypes("V", "Vx");
        auto uBCVals = inputParams.getBCValues("V", "Vx");

#ifndef PLANAR
        auto vBCTypes = inputParams.getBCTypes("V", "Vy");
        auto vBCVals = inputParams.getBCValues("V", "Vy");
#endif

        auto wBCTypes = inputParams.getBCTypes("V", "Vz");
        auto wBCVals = inputParams.getBCValues("V", "Vz");

        std::vector<boundary*> ubList, vbList, wbList;

        for (int i=0; i<nBCs; i++) {
            if (uBCTypes[i])
                ubList.push_back(new neumann(mesh, V.Vx, wn[i], uBCVals[i]));
            else
                ubList.push_back(new dirichlet(mesh, V.Vx, wn[i], uBCVals[i]));

#ifndef PLANAR
            if (vBCTypes[i])
                vbList.push_back(new neumann(mesh, V.Vy, wn[i], vBCVals[i]));
            else
                vbList.push_back(new dirichlet(mesh, V.Vy, wn[i], vBCVals[i]));
#endif

            if (wBCTypes[i])
                wbList.push_back(new neumann(mesh, V.Vz, wn[i], wBCVals[i]));
            else
                wbList.push_back(new dirichlet(mesh, V.Vz, wn[i], wBCVals[i]));
        }

#ifndef PLANAR
        V.uLft = ubList[0];     V.uRgt = ubList[1];
        V.uFrn = ubList[2];     V.uBak = ubList[3];
        V.uBot = ubList[4];     V.uTop = ubList[5];

        V.vLft = vbList[0];     V.vRgt = vbList[1];
        V.vFrn = vbList[2];     V.vBak = vbList[3];
        V.vBot = vbList[4];     V.vTop = vbList[5];

        V.wLft = wbList[0];     V.wRgt = wbList[1];
        V.wFrn = wbList[2];     V.wBak = wbList[3];
        V.wBot = wbList[4];     V.wTop = wbList[5];
#else
        V.uLft = ubList[0];     V.uRgt = ubList[1];
        V.uBot = ubList[2];     V.uTop = ubList[3];

        V.wLft = wbList[0];     V.wRgt = wbList[1];
        V.wBot = wbList[2];     V.wTop = wbList[3];
#endif
    } else {
        if (inputParams.probType == 3) {
            // INFLOW AND OUTFLOW BCS
            V.uLft = new dirichlet(mesh, V.Vx, 0, 1.0);
            V.uRgt = new neumann(mesh, V.Vx, 1, 0.0);
        } else {
            // NO-PENETRATION BCS
            V.uLft = new dirichlet(mesh, V.Vx, 0, 0.0);
            V.uRgt = new dirichlet(mesh, V.Vx, 1, 0.0);
        }

#ifndef PLANAR
        // NO-SLIP BCS
        V.uFrn = new dirichlet(mesh, V.Vx, 2, 0.0);
        V.uBak = new dirichlet(mesh, V.Vx, 3, 0.0);
#endif

        if (inputParams.probType == 1) {
            // NO-SLIP BCS FOR LDC
            V.uBot = new dirichlet(mesh, V.Vx, 4, 0.0);
            V.uTop = new dirichlet(mesh, V.Vx, 5, 1.0);
        } else if (inputParams.probType == 8) {
            // FREE-SLIP BCS
            V.uBot = new neumann(mesh, V.Vx, 4, 0.0);
            V.uTop = new neumann(mesh, V.Vx, 5, 0.0);
        } else {
            // NO-SLIP BCS
            V.uBot = new dirichlet(mesh, V.Vx, 4, 0.0);
            V.uTop = new dirichlet(mesh, V.Vx, 5, 0.0);
        }

#ifndef PLANAR
        if (inputParams.probType == 3) {
            // INFLOW AND OUTFLOW BCS
            V.vLft = new dirichlet(mesh, V.Vy, 0, 0.0);
            V.vRgt = new neumann(mesh, V.Vy, 1, 0.0);
        } else {
            // NO-SLIP BCS
            V.vLft = new dirichlet(mesh, V.Vy, 0, 0.0);
            V.vRgt = new dirichlet(mesh, V.Vy, 1, 0.0);
        }

        // NO-PENETRATION BCS
        V.vFrn = new dirichlet(mesh, V.Vy, 2, 0.0);
        V.vBak = new dirichlet(mesh, V.Vy, 3, 0.0);

        if (inputParams.probType == 8) {
            // FREE-SLIP BCS
            V.vBot = new neumann(mesh, V.Vy, 4, 0.0);
            V.vTop = new neumann(mesh, V.Vy, 5, 0.0);
        } else {
            // NO-SLIP BCS
            V.vBot = new dirichlet(mesh, V.Vy, 4, 0.0);
            V.vTop = new dirichlet(mesh, V.Vy, 5, 0.0);
        }
#endif

        if (inputParams.probType == 3) {
            // INFLOW AND OUTFLOW BCS
            V.wLft = new dirichlet(mesh, V.Vz, 0, 0.0);
            V.wRgt = new neumann(mesh, V.Vz, 1, 0.0);
        } else if (inputParams.probType == 9) {
            // SPECIAL BCs FOR PLUME TEST
            V.wLft = new neumann(mesh, V.Vz, 0, 0.0);
            V.wRgt = new neumann(mesh, V.Vz, 1, 0.0);
        } else {
            // NO-SLIP BCS
            V.wLft = new dirichlet(mesh, V.Vz, 0, 0.0);
            V.wRgt = new dirichlet(mesh, V.Vz, 1, 0.0);
        }

#ifndef PLANAR
        // NO-SLIP BCS
        V.wFrn = new dirichlet(mesh, V.Vz, 2, 0.0);
        V.wBak = new dirichlet(mesh, V.Vz, 3, 0.0);
#endif

        if (inputParams.probType == 9) {
            // SPECIAL BCs FOR PLUME TEST
            V.wBot = new dirichlet(mesh, V.Vz, 4, 0.0);
            V.wTop = new neumann(mesh, V.Vz, 5, 0.0);
        } else {
            // NO-PENETRATION BCS
            V.wBot = new dirichlet(mesh, V.Vz, 4, 0.0);
            V.wTop = new dirichlet(mesh, V.Vz, 5, 0.0);
        }
    }
};


/**
 ********************************************************************************************************************************************
 * \brief   Function to initialize the boundary conditions for pressure
 *
 *          The boundary conditions for all the 6 walls (4 in case of 2D simulations) are initialized here.
 *          Out of the different boundary conditions available in the boundary class,
 *          the appropriate BCs are chosen according to the type of problem being solved.
 ********************************************************************************************************************************************
 */
void hydro::initPBCs() {
    /*******************************************************************************
    * Note for no-slip walls:
    * The initial condition for P satisfies Neumann BC (uniform pressure everywhere
    * including the ghost points).
    * P is updated using pressure correction from pressure-Poisson equation.
    * This pressure correction (from multi-grid solver) obeys Neumann BC.
    * Hence P satisfies Neumann BC at all times implicitly.
    *
    * As a result, theoretically, null BC will work for no-slip walls below.
    * However in this case, a bug crops during restart.
    * The initial condition for P at restart no longer satisfies Neumann BC.
    * This is because the restart file populates only the core of the domain and
    * excludes the ghost points.
    * This causes the solver to diverge badly upon restart.
    * Hence Neumann BC is applied on P below.
    * And this BC is imposed only to help the solver restart without hiccups.
    *******************************************************************************/
    
    if (inputParams.probType == 3) {
        // INFLOW AND OUTFLOW BCS
        P.tLft = new nullBC(mesh, P.F, 0);
        P.tRgt = new neumann(mesh, P.F, 1, 0.0);
    } else {
        // NEUMANN BC FOR NO-SLIP WALLS
        P.tLft = new neumann(mesh, P.F, 0, 0.0);
        P.tRgt = new neumann(mesh, P.F, 1, 0.0);
    }

#ifndef PLANAR
    // NEUMANN BC FOR NO-SLIP WALLS
    P.tFrn = new neumann(mesh, P.F, 2, 0.0);
    P.tBak = new neumann(mesh, P.F, 3, 0.0);
#endif

    // NEUMANN BC FOR NO-SLIP WALLS
    P.tBot = new neumann(mesh, P.F, 4, 0.0);
    P.tTop = new neumann(mesh, P.F, 5, 0.0);
};

