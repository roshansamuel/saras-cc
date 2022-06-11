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


//static std::vector<int> getBLLimits(grid &mesh, std::vector<real> tList);
static void computeDiss(global &gloData, std::vector<real> tList);


void dissipation(global &gloData, std::vector<real> tList) {
    //std::vector<int> blLims = getBLLimits(gridData, tList);

    computeDiss(gloData, tList);
}


/*
static std::vector<int> getBLLimits(grid &mesh, std::vector<real> tList) {
    int ePts = 5;
    real xLim, yLim, zLim;
    std::vector<int> blLims(6, 0);

    vfield V(mesh, "V");
    sfield T(mesh, "T");

    // Fields to be read from HDF5 file are passed to reader class as a vector
    std::vector<field> readFields;

    // Populate the vector with required fields
    readFields.push_back(V.Vx);
    readFields.push_back(V.Vy);
    readFields.push_back(V.Vz);
    readFields.push_back(T.F);

    reader dataReader(mesh);

    // Initialize velocity and temperature boundary conditions
    initVBCs(mesh, V);
    initTBCs(mesh, T);

    // WARNING: These values are temporarily hard-coded.
    // They need to be calculated appropriately to be made general.
    xLim = 0.025;
    yLim = 0.025;
    zLim = 0.05;

    if (mesh.pf) std::cout << xLim << "\t" << yLim << "\t" << zLim << "\t" << ePts << "\n\n";

    // Find thickness of BL at top and bottom plates
    for (unsigned int i=0; i<tList.size(); i++) {
        dataReader.readSolution(tList[i], readFields);

        V.imposeBCs();
        T.imposeBCs();
    }

    return blLims;
}
*/


static void computeDiss(global &gloData, std::vector<real> tList) {
    real d = 1.0;
    real delta = 1.0;
    real Ra, Pr, nu, kappa;
    real epsUNorm, epsTNorm;

    vfield V(gloData.mesh, "V");
    sfield T(gloData.mesh, "T");

    blitz::Array<real, 3> tmpArr1(V.Vx.F.lbound(), V.Vx.F.extent());
    blitz::Array<real, 3> tmpArr2(V.Vx.F.lbound(), V.Vx.F.extent());
    blitz::Array<real, 3> tmpArr3(V.Vx.F.lbound(), V.Vx.F.extent());
    blitz::Array<real, 3> dataArr(V.Vx.F.lbound(), V.Vx.F.extent());

    mpidata dataSync(dataArr, gloData.mesh.rankData);
    dataSync.createSubarrays(dataArr.extent(), gloData.mesh.coreDomain.ubound() + 1, gloData.mesh.padWidths);

    //int intX0 = -1;
    //int intY0 = -1;
    //int intZ0 = -1;

    //int intX1 = gloData.mesh.coreDomain.ubound(0) + 1;
    //int intY1 = gloData.mesh.coreDomain.ubound(1) + 1;
    //int intZ1 = gloData.mesh.coreDomain.ubound(2) + 1;

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

        //dataArr = blitz::sqr(gloData.shift2Wall(V.Vx.F));
        //real uRMS = gloData.simpsonInt(dataArr, gloData.mesh.z, gloData.mesh.y, gloData.mesh.x);
        //uRMS = gloData.volAvgMidPt(dataArr);
        //std::cout << uRMS << std::endl;

        // U_RMS Calculation
        dataArr = blitz::sqr(gloData.shift2Wall(V.Vx.F)) +
                  blitz::sqr(gloData.shift2Wall(V.Vy.F)) +
                  blitz::sqr(gloData.shift2Wall(V.Vz.F));

        //real uRMS = sqrt(gloData.volAvgMidPt(dataArr));
        real uRMS = gloData.simpsonInt(dataArr, gloData.mesh.z, gloData.mesh.y, gloData.mesh.x);
        std::cout << std::setprecision(16) << uRMS << std::endl;
        MPI_Finalize();
        exit(0);

        //if (gloData.mesh.rankData.rank == 0) std::cout << gloData.mesh.z << std::endl;
        //if (gloData.mesh.rankData.rank == 1) std::cout << gloData.mesh.z << std::endl;
        //if (gloData.mesh.rankData.rank == 6) std::cout << gloData.mesh.z << std::endl;
        //if (gloData.mesh.rankData.rank == 7) std::cout << gloData.mesh.z << std::endl;
        //std::cout << dataArr(0, 0, 0) << std::endl;

        // Dissipation normalization factors
        epsUNorm = std::pow(uRMS, 3)/d;
        epsTNorm = uRMS*std::pow(delta, 2)/d;

        // Viscous dissipation
        //if (gloData.mesh.pf) std::cout << gloData.mesh.dXi << std::endl;
        //if (gloData.mesh.pf) std::cout << V.Vx.F(blitz::Range(29, 33, 1), 5, 5) << std::endl;
        //if (gloData.mesh.pf) std::cout << gloData.mesh.x(blitz::Range(29, 33, 1)) << std::endl;
        //if (gloData.mesh.pf) std::cout << dataArr(blitz::Range(-2, 2, 1), 5, 5) << std::endl;
        V.derVx.calcDerivative1_x(tmpArr1);

        V.derVy.calcDerivative1_y(tmpArr2);
        V.derVz.calcDerivative1_z(tmpArr3);
        dataArr = 2.0*(blitz::sqr(tmpArr1) + blitz::sqr(tmpArr2) + blitz::sqr(tmpArr3));

        //if (gloData.mesh.rankData.rank == 0) std::cout << dataArr(5, 5, blitz::Range(-2, 1, 1)) << std::endl;
        //if (gloData.mesh.rankData.rank == 7) std::cout << dataArr(5, 5, blitz::Range(30, 33, 1)) << std::endl;

        //if (gloData.mesh.rankData.rank == 0) std::cout << dataArr(5, 5, blitz::Range(30, 33, 1)) << std::endl;
        //if (gloData.mesh.rankData.rank == 1) std::cout << dataArr(5, 5, blitz::Range(-2, 1, 1)) << std::endl;
        dataSync.syncAll();

        //if (gloData.mesh.rankData.rank == 0) std::cout << dataArr(5, 5, blitz::Range(-2, 1, 1)) << std::endl;
        //if (gloData.mesh.rankData.rank == 7) std::cout << dataArr(5, 5, blitz::Range(30, 33, 1)) << std::endl;

        //if (gloData.mesh.rankData.rank == 0) std::cout << dataArr(5, 5, blitz::Range(30, 33, 1)) << std::endl;
        //if (gloData.mesh.rankData.rank == 1) std::cout << dataArr(5, 5, blitz::Range(-2, 1, 1)) << std::endl;
        //MPI_Finalize();
        //exit(0);

        V.derVx.calcDerivative1_y(tmpArr1);
        V.derVy.calcDerivative1_x(tmpArr2);
        dataArr += blitz::sqr(tmpArr1 + tmpArr2);

        V.derVx.calcDerivative1_z(tmpArr1);
        V.derVz.calcDerivative1_x(tmpArr2);
        dataArr += blitz::sqr(tmpArr1 + tmpArr2);

        V.derVy.calcDerivative1_z(tmpArr1);
        V.derVz.calcDerivative1_y(tmpArr2);
        dataArr += blitz::sqr(tmpArr1 + tmpArr2);

        dataArr *= nu;
        real epsU = gloData.volAvgMidPt(dataArr)/epsUNorm;

        // Thermal dissipation
        T.derS.calcDerivative1_x(tmpArr1);
        T.derS.calcDerivative1_y(tmpArr2);
        T.derS.calcDerivative1_z(tmpArr3);

        dataArr = kappa*(blitz::sqr(tmpArr1) + blitz::sqr(tmpArr2) + blitz::sqr(tmpArr3));
        real epsT = gloData.volAvgMidPt(dataArr)/epsTNorm;

        if (gloData.mesh.pf) std::cout << std::setprecision(16) << epsU << "\t" << epsT << "\n";

        MPI_Finalize();
        exit(0);
    }
}
