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
/*! \file dissipation.cc
 *
 *  \brief Definitions of functions to perform dissipation calculations
 *  \sa postprocess.h
 *  \author Roshan Samuel
 *  \date May 2022
 *  \copyright New BSD License
 *
 ********************************************************************************************************************************************
 */

#include <iostream>
#include "postprocess.h"


static std::vector<int> getBLLimits(global &gloData, std::vector<real> tList);

static std::vector<int> getRoughLim(global &gloData);

static void computeDiss(global &gloData, std::vector<real> tList);

static void computeBulkDiss(global &gloData, std::vector<real> tList);


void dissipation(global &gloData, std::vector<real> tList) {
    computeBulkDiss(gloData, tList);
    //computeDiss(gloData, tList);
}


static std::vector<int> getRoughLim(global &gloData) {
    real xLim, yLim, zLim;
    std::vector<int> roughLim(6, 0);

    // These values need to be calculated appropriately to be made general.
    xLim = gloData.mesh.xLen / (gloData.mesh.rankData.npX * 5);
    yLim = gloData.mesh.yLen / (gloData.mesh.rankData.npY * 5);
    zLim = gloData.mesh.zLen / (gloData.mesh.rankData.npZ * 5);

    for (int i=0; i<gloData.mesh.inputParams.Nx; i++) {
        if (gloData.mesh.xGlobal(i) > xLim) {
            roughLim[0] = i-1;
            roughLim[1] = gloData.mesh.inputParams.Nx - i;
            break;
        }
    }

    for (int i=0; i<gloData.mesh.inputParams.Ny; i++) {
        if (gloData.mesh.yGlobal(i) > yLim) {
            roughLim[2] = i-1;
            roughLim[3] = gloData.mesh.inputParams.Ny - i;
            break;
        }
    }

    for (int i=0; i<gloData.mesh.inputParams.Nz; i++) {
        if (gloData.mesh.zGlobal(i) > zLim) {
            roughLim[4] = i-1;
            roughLim[5] = gloData.mesh.inputParams.Nz - i;
            break;
        }
    }

    return roughLim;
}


static std::vector<int> getBLLimits(global &gloData, std::vector<real> tList) {
    std::vector<int> blLims(6, 0);
    blitz::TinyVector<int, 2> lb, ub;

    vfield V(gloData.mesh, "V");
    sfield T(gloData.mesh, "T");

    blitz::Array<real, 3> dataArr(V.Vx.F.lbound(), V.Vx.F.extent());
    blitz::Array<real, 2> xySlice, yzSlice, zxSlice;

    mpidata mpiHandle(gloData.mesh.rankData);
    mpiHandle.createSubarrays(dataArr.extent(), gloData.mesh.coreDomain.ubound() + 1, gloData.mesh.padWidths);

    // Fields to be read from HDF5 file are passed to reader class as a vector
    std::vector<field> readFields;

    // Populate the vector with required fields
    readFields.push_back(V.Vx);
    readFields.push_back(V.Vy);
    readFields.push_back(V.Vz);
    readFields.push_back(T.F);

    reader dataReader(gloData.mesh);

    // Initialize velocity and temperature boundary conditions
    gloData.initVBCs(V);
    gloData.initTBCs(T);

    blLims = getRoughLim(gloData);

    // Find thickness of BL at top and bottom plates
    /* WARNING: Temporarily ignoring BL thickness and setting bulk to xLim, yLim, zLim times the wall-normal length
    for (unsigned int i=0; i<tList.size(); i++) {
        dataReader.readSolution(tList[i], readFields);

        V.imposeBCs();
        T.imposeBCs();

        // U_RMS Calculation
        dataArr = gloData.shift2Wall(V.Vx.F)(5, blitz::Range::all(), blitz::Range::all());
        std::cout << dataArr.shape() << std::endl;
        MPI_Finalize();
        exit(0);
    }
    */


    return blLims;
}


static void computeDiss(global &gloData, std::vector<real> tList) {
    real d = 1.0;
    real delta = 1.0;
    real Ra, Pr, nu, kappa;
    real epsUNorm, epsTNorm;
    real totalVol = gloData.mesh.xLen * gloData.mesh.yLen * gloData.mesh.zLen;

    vfield V(gloData.mesh, "V");
    sfield T(gloData.mesh, "T");

    blitz::Array<real, 3> tmpArr1(V.Vx.F.lbound(), V.Vx.F.extent());
    blitz::Array<real, 3> tmpArr2(V.Vx.F.lbound(), V.Vx.F.extent());
    blitz::Array<real, 3> tmpArr3(V.Vx.F.lbound(), V.Vx.F.extent());
    blitz::Array<real, 3> dataArr(V.Vx.F.lbound(), V.Vx.F.extent());

    mpidata mpiHandle(gloData.mesh.rankData);
    mpiHandle.createSubarrays(dataArr.extent(), gloData.mesh.coreDomain.ubound() + 1, gloData.mesh.padWidths);

    // Fields to be read from HDF5 file are passed to reader class as a vector
    std::vector<field> readFields;

    // Populate the vector with required fields
    readFields.push_back(V.Vx);
    readFields.push_back(V.Vy);
    readFields.push_back(V.Vz);
    readFields.push_back(T.F);

    reader dataReader(gloData.mesh);

    // Initialize velocity and temperature boundary conditions
    gloData.initVBCs(V);
    gloData.initTBCs(T);

    Ra = gloData.mesh.inputParams.Ra;
    Pr = gloData.mesh.inputParams.Pr;
    nu = sqrt(Pr/Ra);
    kappa = 1.0/sqrt(Pr*Ra);

    if (gloData.mesh.pf) std::cout << "nu = " << nu << "\tkappa = " << kappa << "\n";

    for (unsigned int i=0; i<tList.size(); i++) {
        dataReader.readSolution(tList[i], readFields);

        V.imposeBCs();
        T.imposeBCs();

        // U_RMS Calculation
        dataArr = blitz::sqr(gloData.shift2Wall(V.Vx.F)) +
                  blitz::sqr(gloData.shift2Wall(V.Vy.F)) +
                  blitz::sqr(gloData.shift2Wall(V.Vz.F));

        real uRMS = std::sqrt(gloData.simpsonInt(dataArr, gloData.mesh.z, gloData.mesh.y, gloData.mesh.x)/totalVol);

        // Dissipation normalization factors
        epsUNorm = std::pow(uRMS, 3)/d;
        epsTNorm = uRMS*std::pow(delta, 2)/d;

        // Viscous dissipation
        V.derVx.calcDerivative1_x(tmpArr1);     mpiHandle.syncAll(tmpArr1);
        V.derVy.calcDerivative1_y(tmpArr2);     mpiHandle.syncAll(tmpArr2);
        V.derVz.calcDerivative1_z(tmpArr3);     mpiHandle.syncAll(tmpArr3);
        dataArr = 2.0*(blitz::sqr(gloData.shift2Wall(tmpArr1)) +
                       blitz::sqr(gloData.shift2Wall(tmpArr2)) +
                       blitz::sqr(gloData.shift2Wall(tmpArr3)));

        V.derVx.calcDerivative1_y(tmpArr1);     mpiHandle.syncAll(tmpArr1);
        V.derVy.calcDerivative1_x(tmpArr2);     mpiHandle.syncAll(tmpArr2);
        dataArr += blitz::sqr(gloData.shift2Wall(tmpArr1) + gloData.shift2Wall(tmpArr2));

        V.derVx.calcDerivative1_z(tmpArr1);     mpiHandle.syncAll(tmpArr1);
        V.derVz.calcDerivative1_x(tmpArr2);     mpiHandle.syncAll(tmpArr2);
        dataArr += blitz::sqr(gloData.shift2Wall(tmpArr1) + gloData.shift2Wall(tmpArr2));

        V.derVy.calcDerivative1_z(tmpArr1);     mpiHandle.syncAll(tmpArr1);
        V.derVz.calcDerivative1_y(tmpArr2);     mpiHandle.syncAll(tmpArr2);
        dataArr += blitz::sqr(gloData.shift2Wall(tmpArr1) + gloData.shift2Wall(tmpArr2));

        dataArr *= nu;
        tmpArr1 = blitz::pow((nu*nu*nu)/dataArr, 0.25);
        real epsU = gloData.simpsonInt(dataArr, gloData.mesh.z, gloData.mesh.y, gloData.mesh.x)/totalVol;
        real etaL = gloData.simpsonInt(tmpArr1, gloData.mesh.z, gloData.mesh.y, gloData.mesh.x)/totalVol;

        // Thermal dissipation
        T.derS.calcDerivative1_x(tmpArr1);     mpiHandle.syncAll(tmpArr1);
        T.derS.calcDerivative1_y(tmpArr2);     mpiHandle.syncAll(tmpArr2);
        T.derS.calcDerivative1_z(tmpArr3);     mpiHandle.syncAll(tmpArr3);

        dataArr = kappa*(blitz::sqr(gloData.shift2Wall(tmpArr1)) +
                         blitz::sqr(gloData.shift2Wall(tmpArr2)) +
                         blitz::sqr(gloData.shift2Wall(tmpArr3)));
        real epsT = gloData.simpsonInt(dataArr, gloData.mesh.z, gloData.mesh.y, gloData.mesh.x)/totalVol;

        if (gloData.mesh.pf) std::cout << std::setprecision(16) << nu << "\t" << etaL << "\t" << epsU << "\t" << epsT << "\t" << epsU/epsUNorm << "\t" << epsT/epsTNorm << "\n";
    }
}


static void computeBulkDiss(global &gloData, std::vector<real> tList) {
    real d = 1.0;
    real delta = 1.0;
    real Ra, Pr, nu, kappa;
    real epsUNorm, epsTNorm;
    real totalVol = gloData.mesh.xLen * gloData.mesh.yLen * gloData.mesh.zLen;
    std::vector<int> blLims = getBLLimits(gloData, tList);

    std::cout << blLims[0] << "\t" << blLims[1] << "\t" << blLims[2] << "\t" << blLims[3] << "\t" << blLims[4] << "\t" << blLims[5] << std::endl;
    MPI_Finalize();
    exit(0);

    vfield V(gloData.mesh, "V");
    sfield T(gloData.mesh, "T");

    blitz::Array<real, 3> tmpArr1(V.Vx.F.lbound(), V.Vx.F.extent());
    blitz::Array<real, 3> tmpArr2(V.Vx.F.lbound(), V.Vx.F.extent());
    blitz::Array<real, 3> tmpArr3(V.Vx.F.lbound(), V.Vx.F.extent());
    blitz::Array<real, 3> dataArr(V.Vx.F.lbound(), V.Vx.F.extent());

    mpidata mpiHandle(gloData.mesh.rankData);
    mpiHandle.createSubarrays(dataArr.extent(), gloData.mesh.coreDomain.ubound() + 1, gloData.mesh.padWidths);

    // Fields to be read from HDF5 file are passed to reader class as a vector
    std::vector<field> readFields;

    // Populate the vector with required fields
    readFields.push_back(V.Vx);
    readFields.push_back(V.Vy);
    readFields.push_back(V.Vz);
    readFields.push_back(T.F);

    reader dataReader(gloData.mesh);

    // Initialize velocity and temperature boundary conditions
    gloData.initVBCs(V);
    gloData.initTBCs(T);

    Ra = gloData.mesh.inputParams.Ra;
    Pr = gloData.mesh.inputParams.Pr;
    nu = sqrt(Pr/Ra);
    kappa = 1.0/sqrt(Pr*Ra);

    if (gloData.mesh.pf) std::cout << "nu = " << nu << "\tkappa = " << kappa << "\n";

    for (unsigned int i=0; i<tList.size(); i++) {
        dataReader.readSolution(tList[i], readFields);

        V.imposeBCs();
        T.imposeBCs();

        // U_RMS Calculation
        dataArr = blitz::sqr(gloData.shift2Wall(V.Vx.F)) +
                  blitz::sqr(gloData.shift2Wall(V.Vy.F)) +
                  blitz::sqr(gloData.shift2Wall(V.Vz.F));

        real uRMS = std::sqrt(gloData.simpsonInt(dataArr, gloData.mesh.z, gloData.mesh.y, gloData.mesh.x)/totalVol);

        // Dissipation normalization factors
        epsUNorm = std::pow(uRMS, 3)/d;
        epsTNorm = uRMS*std::pow(delta, 2)/d;

        // Viscous dissipation
        V.derVx.calcDerivative1_x(tmpArr1);     mpiHandle.syncAll(tmpArr1);
        V.derVy.calcDerivative1_y(tmpArr2);     mpiHandle.syncAll(tmpArr2);
        V.derVz.calcDerivative1_z(tmpArr3);     mpiHandle.syncAll(tmpArr3);
        dataArr = 2.0*(blitz::sqr(gloData.shift2Wall(tmpArr1)) +
                       blitz::sqr(gloData.shift2Wall(tmpArr2)) +
                       blitz::sqr(gloData.shift2Wall(tmpArr3)));

        V.derVx.calcDerivative1_y(tmpArr1);     mpiHandle.syncAll(tmpArr1);
        V.derVy.calcDerivative1_x(tmpArr2);     mpiHandle.syncAll(tmpArr2);
        dataArr += blitz::sqr(gloData.shift2Wall(tmpArr1) + gloData.shift2Wall(tmpArr2));

        V.derVx.calcDerivative1_z(tmpArr1);     mpiHandle.syncAll(tmpArr1);
        V.derVz.calcDerivative1_x(tmpArr2);     mpiHandle.syncAll(tmpArr2);
        dataArr += blitz::sqr(gloData.shift2Wall(tmpArr1) + gloData.shift2Wall(tmpArr2));

        V.derVy.calcDerivative1_z(tmpArr1);     mpiHandle.syncAll(tmpArr1);
        V.derVz.calcDerivative1_y(tmpArr2);     mpiHandle.syncAll(tmpArr2);
        dataArr += blitz::sqr(gloData.shift2Wall(tmpArr1) + gloData.shift2Wall(tmpArr2));

        dataArr *= nu;
        tmpArr1 = blitz::pow((nu*nu*nu)/dataArr, 0.25);
        real epsU = gloData.simpsonInt(dataArr, gloData.mesh.z, gloData.mesh.y, gloData.mesh.x)/totalVol;
        real etaL = gloData.simpsonInt(tmpArr1, gloData.mesh.z, gloData.mesh.y, gloData.mesh.x)/totalVol;

        // Thermal dissipation
        T.derS.calcDerivative1_x(tmpArr1);     mpiHandle.syncAll(tmpArr1);
        T.derS.calcDerivative1_y(tmpArr2);     mpiHandle.syncAll(tmpArr2);
        T.derS.calcDerivative1_z(tmpArr3);     mpiHandle.syncAll(tmpArr3);

        dataArr = kappa*(blitz::sqr(gloData.shift2Wall(tmpArr1)) +
                         blitz::sqr(gloData.shift2Wall(tmpArr2)) +
                         blitz::sqr(gloData.shift2Wall(tmpArr3)));
        real epsT = gloData.simpsonInt(dataArr, gloData.mesh.z, gloData.mesh.y, gloData.mesh.x)/totalVol;

        if (gloData.mesh.pf) std::cout << std::setprecision(16) << etaL << "\t" << epsU << "\t" << epsT << "\t" << epsU/epsUNorm << "\t" << epsT/epsTNorm << "\n";
    }
}
