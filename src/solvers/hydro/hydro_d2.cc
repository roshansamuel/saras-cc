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
/*! \file hydro_d2.cc
 *
 *  \brief Definitions of functions for 2D computations with the hydro class.
 *  \sa hydro.h
 *  \author Roshan Samuel
 *  \date Nov 2019
 *  \copyright New BSD License
 *
 ********************************************************************************************************************************************
 */

#include "hydro.h"
#include "initial.h"

/**
 ********************************************************************************************************************************************
 * \brief   Constructor of the hydro_d2 class derived from the base hydro class
 *
 *          The constructor passes its arguments to the base hydro class and then initializes all the scalar and
 *          vector fields necessary for solving the NS equations.
 *          Based on the problem type specified by the user in the parameters file, and stored by the \ref parser class as
 *          \ref parser#probType "probType", the appropriate boundary conditions are specified.
 *
 * \param   mesh is a const reference to the global data contained in the grid class
 * \param   solParam is a const reference to the user-set parameters contained in the parser class
 ********************************************************************************************************************************************
 */
hydro_d2::hydro_d2(const grid &mesh, const parser &solParam, parallel &mpiParam):
            hydro(mesh, solParam, mpiParam)
{
    // INITIALIZE VARIABLES
    if (inputParams.restartFlag) {
        // Fields to be read from HDF5 file are passed to reader class as a vector
        std::vector<field> readFields;

        // Populate the vector with required fields
        readFields.push_back(V.Vx);
        readFields.push_back(V.Vz);
        readFields.push_back(P.F);

        // Initialize reader object
        reader dataReader(mesh, readFields);

        time = dataReader.readData();

        // Abort if this time is greater than the final time specified by the user
        if (time >= inputParams.tMax) {
            if (mesh.rankData.rank == 0) {
                std::cout << "ERROR: Restart file is starting from a point beyond the final time specified. Aborting" << std::endl;
            }
            MPI_Finalize();
            exit(0);
        }
    } else {
        time = 0.0;

        // INITIALIZE PRESSURE TO 1.0 THROUGHOUT THE DOMAIN
        P = 1.0;

        // INITIALIZE VELOCITY FIELD
        initial *initCond;
        switch (inputParams.icType) {
            case 0: initCond = new zeroInitial(mesh);
                break;
            case 1: initCond = new taylorGreen(mesh);
                break;
            case 2: initCond = new channelSine(mesh);
                break;
            case 3: initCond = new uniformRandom(mesh);
                break;
            case 4: initCond = new parabolicRandom(mesh);
                break;
            case 5: initCond = new sineRandom(mesh);
                break;
            default: initCond = new zeroInitial(mesh);
        }
        initCond->initializeField(V);
    }

    checkPeriodic();

    // Initialize velocity and pressure boundary conditions
    initVBCs();
    initPBCs();

    // Initialize velocity forcing field
    initVForcing();

    // Impose boundary conditions on velocity and pressure fields
    V.imposeBCs();
    P.imposeBCs();
}


void hydro_d2::solvePDE() {
    real fwTime, prTime, rsTime;

    // Set dt equal to input time step
    dt = inputParams.tStp;

    // Fields to be written into HDF5 file are passed to writer class as a vector
    std::vector<field> writeFields;

    // Populate the vector with required fields
    writeFields.push_back(V.Vx);
    writeFields.push_back(V.Vz);
    writeFields.push_back(P.F);

    // Initialize writer object
    writer dataWriter(mesh, writeFields);

    // Initialize probes
    if (inputParams.readProbes) {
        dataProbe = new probes(mesh, writeFields);
    }

    // Output file and I/O writer to write time-series of various variables
    tseries tsWriter(mesh, V, P, time, dt);

    // Initialize semi-implicit Euler-CN time-stepping method
    ivpSolver = new eulerCN_d2(mesh, time, dt, tsWriter, V, P);

    // FILE WRITING TIME
    fwTime = time;

    // FIELD PROBING TIME
    prTime = time;

    // RESTART FILE WRITING TIME
    rsTime = time;

    timeStepCount = 0;

    // WRITE THE HEADERS FOR TIME-SERIES IN I/O AND IN FILE
    tsWriter.writeTSHeader();

    // COMPUTE ENERGY AND DIVERGENCE FOR THE INITIAL CONDITION
    tsWriter.writeTSData();

    // WRITE DATA AT t = 0 OR INCREMENT INTERVAL IF RESTARTING
    if (inputParams.restartFlag) {
        int tCount, fCount;

        tCount = int(time/inputParams.tStp);

        fCount = int(inputParams.fwInt/inputParams.tStp);
        fwTime = roundNum(tCount, fCount)*inputParams.tStp;

        fCount = int(inputParams.prInt/inputParams.tStp);
        prTime = roundNum(tCount, fCount)*inputParams.tStp;

        fCount = int(inputParams.rsInt/inputParams.tStp);
        rsTime = roundNum(tCount, fCount)*inputParams.tStp;
    }

    switch (inputParams.solnFormat) {
        case 1: dataWriter.writeSolution(time);
            break;
        case 2: dataWriter.writeTarang(time);
            break;
        default: dataWriter.writeSolution(time);
    }
    fwTime += inputParams.fwInt;

    if (inputParams.readProbes) {
        dataProbe->probeData(time);
        prTime += inputParams.prInt;
    }

    rsTime += inputParams.rsInt;

    // TIME-INTEGRATION LOOP
    while (true) {
        // MAIN FUNCTION CALLED IN EACH LOOP TO UPDATE THE FIELDS AT EACH TIME-STEP
        ivpSolver->timeAdvance(V, P);

        if (inputParams.useCFL) {
            V.computeTStp(dt);
            if (dt > inputParams.tStp) {
                dt = inputParams.tStp;
            }
        }

        timeStepCount += 1;
        time += dt;

        if (timeStepCount % inputParams.ioCnt == 0) {
            tsWriter.writeTSData();
        }

        if (inputParams.readProbes and std::abs(prTime - time) < 0.5*dt) {
            dataProbe->probeData(time);
            prTime += inputParams.prInt;
        }

        if (std::abs(fwTime - time) < 0.5*dt) {
            switch (inputParams.solnFormat) {
                case 1: dataWriter.writeSolution(time);
                    break;
                case 2: dataWriter.writeTarang(time);
                    break;
                default: dataWriter.writeSolution(time);
            }
            fwTime += inputParams.fwInt;
        }

        if (std::abs(rsTime - time) < 0.5*dt) {
            dataWriter.writeRestart(time);
            rsTime += inputParams.rsInt;
        }

        if (std::abs(inputParams.tMax - time) < 0.5*dt) {
            break;
        }
    }
}


real hydro_d2::testPeriodic() {
    int iY = 0;
    real xCoord = 0.0;
    real zCoord = 0.0;

    plainvf nseRHS(mesh, V);
    nseRHS = 0.0;
    V = 0.0;

    for (int i=V.Vx.F.lbound(0); i <= V.Vx.F.ubound(0); i++) {
        for (int k=V.Vx.F.lbound(2); k <= V.Vx.F.ubound(2); k++) {
            V.Vx.F(i, 0, k) = sin(2.0*M_PI*mesh.x(i)/mesh.xLen)*
                              cos(2.0*M_PI*mesh.z(k)/mesh.zLen);
            nseRHS.Vx(i, 0, k) = V.Vx.F(i, 0, k);
        }
    }

    for (int i=V.Vz.F.lbound(0); i <= V.Vz.F.ubound(0); i++) {
        for (int k=V.Vz.F.lbound(2); k <= V.Vz.F.ubound(2); k++) {
            V.Vz.F(i, 0, k) = -cos(2.0*M_PI*mesh.x(i)/mesh.xLen)*
                               sin(2.0*M_PI*mesh.z(k)/mesh.zLen);
            nseRHS.Vz(i, 0, k) = V.Vz.F(i, 0, k);
        }
    }

    // EXPECTED VALUES IN THE PAD REGIONS OF Vx IF DATA TRANSFER HAPPENS WITH NO HITCH
    // X-VELOCITY IN LEFT AND RIGHT PADS
    for (int iX = 1; iX <= mesh.padWidths(0); iX++) {
        for (int iZ = V.Vx.fCore.lbound(2); iZ <= V.Vx.fCore.ubound(2); iZ += 1) {
            xCoord = mesh.x(V.Vx.fCore.lbound(0)) - (mesh.x(V.Vx.fCore.lbound(0) + iX) - mesh.x(V.Vx.fCore.lbound(0)));
            nseRHS.Vx(V.Vx.fCore.lbound(0) - iX, iY, iZ) = sin(2.0*M_PI*xCoord/mesh.xLen)*
                                                           cos(2.0*M_PI*mesh.z(iZ)/mesh.zLen);

            xCoord = mesh.x(V.Vx.fCore.ubound(0)) + (mesh.x(V.Vx.fCore.ubound(0)) - mesh.x(V.Vx.fCore.ubound(0) - iX));
            nseRHS.Vx(V.Vx.fCore.ubound(0) + iX, iY, iZ) = sin(2.0*M_PI*xCoord/mesh.xLen)*
                                                           cos(2.0*M_PI*mesh.z(iZ)/mesh.zLen);
        }
    }

    // X-VELOCITY IN BOTTOM AND TOP PADS
    for (int iZ = 1; iZ <= mesh.padWidths(2); iZ++) {
        for (int iX = V.Vx.fCore.lbound(0); iX <= V.Vx.fCore.ubound(0); iX += 1) {
            zCoord = mesh.z(V.Vx.fCore.lbound(2)) - (mesh.z(V.Vx.fCore.lbound(2) + iZ) - mesh.z(V.Vx.fCore.lbound(2)));
            nseRHS.Vx(iX, iY, V.Vx.fCore.lbound(2) - iZ) = sin(2.0*M_PI*mesh.x(iX)/mesh.xLen)*
                                                           cos(2.0*M_PI*zCoord/mesh.zLen);

            zCoord = mesh.z(V.Vx.fCore.ubound(2)) + (mesh.z(V.Vx.fCore.ubound(2)) - mesh.z(V.Vx.fCore.ubound(2) - iZ));
            nseRHS.Vx(iX, iY, V.Vx.fCore.ubound(2) + iZ) = sin(2.0*M_PI*mesh.x(iX)/mesh.xLen)*
                                                           cos(2.0*M_PI*zCoord/mesh.zLen);
        }
    }

    // Z-VELOCITY IN LEFT AND RIGHT PADS
    for (int iX = 1; iX <= mesh.padWidths(0); iX++) {
        for (int iZ = V.Vz.fCore.lbound(2); iZ <= V.Vz.fCore.ubound(2); iZ += 1) {
            xCoord = mesh.x(V.Vz.fCore.lbound(0)) - (mesh.x(V.Vz.fCore.lbound(0) + iX) - mesh.x(V.Vz.fCore.lbound(0)));
            nseRHS.Vz(V.Vz.fCore.lbound(0) - iX, iY, iZ) = -cos(2.0*M_PI*xCoord/mesh.xLen)*
                                                            sin(2.0*M_PI*mesh.z(iZ)/mesh.zLen);

            xCoord = mesh.x(V.Vz.fCore.ubound(0)) + (mesh.x(V.Vz.fCore.ubound(0)) - mesh.x(V.Vz.fCore.ubound(0) - iX));
            nseRHS.Vz(V.Vz.fCore.ubound(0) + iX, iY, iZ) = -cos(2.0*M_PI*xCoord/mesh.xLen)*
                                                            sin(2.0*M_PI*mesh.z(iZ)/mesh.zLen);
        }
    }

    // Z-VELOCITY IN BOTTOM AND TOP PADS
    for (int iZ = 1; iZ <= mesh.padWidths(2); iZ++) {
        for (int iX = V.Vz.fCore.lbound(0); iX <= V.Vz.fCore.ubound(0); iX += 1) {
            zCoord = mesh.z(V.Vz.fCore.lbound(2)) - (mesh.z(V.Vz.fCore.lbound(2) + iZ) - mesh.z(V.Vz.fCore.lbound(2)));
            nseRHS.Vz(iX, iY, V.Vz.fCore.lbound(2) - iZ) = -cos(2.0*M_PI*mesh.x(iX)/mesh.xLen)*
                                                            sin(2.0*M_PI*zCoord/mesh.zLen);

            zCoord = mesh.z(V.Vz.fCore.ubound(2)) + (mesh.z(V.Vz.fCore.ubound(2)) - mesh.z(V.Vz.fCore.ubound(2) - iZ));
            nseRHS.Vz(iX, iY, V.Vz.fCore.ubound(2) + iZ) = -cos(2.0*M_PI*mesh.x(iX)/mesh.xLen)*
                                                            sin(2.0*M_PI*zCoord/mesh.zLen);
        }
    }

    V.imposeBCs();

    V -= nseRHS;

    return std::max(blitz::max(fabs(V.Vx.F)), blitz::max(fabs(V.Vz.F)));
}

hydro_d2::~hydro_d2() { }
