/********************************************************************************************************************************************
 * Saras
 * 
 * Copyright (C) 2022, Roshan J. Samuel
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
/*! \file wallModel.cc
 *
 *  \brief Definitions for functions of LES wall-model
 *  \sa les.h
 *  \date May 2022
 *  \copyright New BSD License
 *
 ********************************************************************************************************************************************
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "les.h"

/**
 ********************************************************************************************************************************************
 * \brief   Constructor of the wall-model class
 *
 *          The constructor initializes the base les class using part of the arguments supplied to it.
 *          The value of viscous diffusion coefficient, denoted by nu, is also set in the initialization list.
 *          All the blitz arrays and scalar fields necessary to compute velocity gradient tensor, stress tensor, etc.
 *          are initialized in the constructor.
 *
 * \param   mesh is a const reference to the global data contained in the grid class.
 * \param   kDiff is a const reference to scalar value denoting viscous dissipation
 ********************************************************************************************************************************************
 */
wallModel::wallModel(const grid &mesh, const int bcWall, const real &kDiff): wallNum(bcWall), mesh(mesh), nu(kDiff) {
    real nwSpacing = 0;
    blitz::TinyVector<int, 3> flBnd, fuBnd, fSize;

    fSize = mesh.fullDomain.ubound() - mesh.fullDomain.lbound() + 1;
    flBnd = mesh.fullDomain.lbound();
    fuBnd = mesh.fullDomain.ubound();

    // By default, rankFlag is true. i.e., the BC will be applied on all sub-domains.
    // This has to be changed appropriately.
    rankFlag = true;

    // By default, shiftVal is 1. i.e., the BC will be applied correctly only on the left wall along a given direction.
    // This has to be changed appropriately for the wall on the other side.
    shiftVal = 1;

    // Update variables for the left wall (along X)
    if (wallNum == 0) {
        rankFlag = mesh.rankData.xRank == 0;
        nwSpacing = mesh.xGlobal(0);
        wInd = mesh.coreDomain.lbound(0);
    }
    // Update variables for the right wall (along X)
    if (wallNum == 1) {
        rankFlag = mesh.rankData.xRank == mesh.rankData.npX - 1;
        nwSpacing = mesh.xLen - mesh.xGlobal(mesh.globalSize(0) - 1);
        shiftVal = -1;
        wInd = mesh.coreDomain.ubound(0);
    }

    // Update variables for the front wall (along Y)
    if (wallNum == 2) {
        rankFlag = mesh.rankData.yRank == 0;
        nwSpacing = mesh.yGlobal(0);
        wInd = mesh.coreDomain.lbound(1);
    }
    // Update variables for the back wall (along Y)
    if (wallNum == 3) {
        rankFlag = mesh.rankData.yRank == mesh.rankData.npY - 1;
        nwSpacing = mesh.yLen - mesh.yGlobal(mesh.globalSize(1) - 1);
        shiftVal = -1;
        wInd = mesh.coreDomain.ubound(1);
    }

    // Update variables for the bottom wall (along Z)
    if (wallNum == 4) {
        rankFlag = mesh.rankData.zRank == 0;
        nwSpacing = mesh.zGlobal(0);
        wInd = mesh.coreDomain.lbound(2);
    }
    // Update variables for the top wall (along Z)
    if (wallNum == 5) {
        rankFlag = mesh.rankData.zRank == mesh.rankData.npZ - 1;
        nwSpacing = mesh.zLen - mesh.zGlobal(mesh.globalSize(2) - 1);
        shiftVal = -1;
        wInd = mesh.coreDomain.ubound(2);
    }

    // Calculate height of virtual wall from physical no-slip wall
    // Factor 2 appears since the solver is cell-centered
    h0 = mesh.inputParams.vwHeight*nwSpacing*2;

    // Calculate height of first mesh point from physical no-slip wall
    // The sign changes appropriately according to direction of shift
    bc_h = double(shiftVal)*(h0 + nwSpacing);

    // This value is set as 11.0 in all instances of WMLES reported so far in literature
    h_nu_plus = 11.0;

    // Find the dimension perpendicular to which wall-model is being applied (X -> 0, Y -> 1, Z -> 2) using the wallNum
    shiftDim = (int) wallNum/2;

    // Set size and lbounds of the wall slice according to the value of shiftDim
    switch (shiftDim) {
        case 0:
            dSize(0) = fSize(1);        dSize(1) = fSize(2);
            dlBnd(0) = flBnd(1);        dlBnd(1) = flBnd(2);
            duBnd(0) = fuBnd(1);        duBnd(1) = fuBnd(2);

            break;
        case 1:
            dSize(0) = fSize(0);        dSize(1) = fSize(2);
            dlBnd(0) = flBnd(0);        dlBnd(1) = flBnd(2);
            duBnd(0) = fuBnd(0);        duBnd(1) = fuBnd(2);

            break;
        case 2:
            dSize(0) = fSize(0);        dSize(1) = fSize(1);
            dlBnd(0) = flBnd(0);        dlBnd(1) = flBnd(1);
            duBnd(0) = fuBnd(0);        duBnd(1) = fuBnd(1);

            break;
    }

    // Resize all arrays using the size and lbounds of wall slice
    q.resize(dSize);            q.reindexSelf(dlBnd);
    K0.resize(dSize);           K0.reindexSelf(dlBnd);
    eta0.resize(dSize);         eta0.reindexSelf(dlBnd);

    Tii.resize(dSize);          Tii.reindexSelf(dlBnd);
    Tjj.resize(dSize);          Tjj.reindexSelf(dlBnd);
    Tij.resize(dSize);          Tij.reindexSelf(dlBnd);

    vi.resize(dSize);           vi.reindexSelf(dlBnd);
    vj.resize(dSize);           vj.reindexSelf(dlBnd);
    vii.resize(dSize);          vii.reindexSelf(dlBnd);
    vjj.resize(dSize);          vjj.reindexSelf(dlBnd);
    vij.resize(dSize);          vij.reindexSelf(dlBnd);

    bcU.resize(dSize);          bcU.reindexSelf(dlBnd);
    bcV.resize(dSize);          bcV.reindexSelf(dlBnd);
    bcW.resize(dSize);          bcW.reindexSelf(dlBnd);

    eta0temp.resize(dSize);     eta0temp.reindexSelf(dlBnd);
}


/**
 ********************************************************************************************************************************************
 * \brief   Function to compute and add contribution from LES model to the RHS of NSE and temperature equation
 *
 *          This function is called when both hydrodynamics and scalar equations are being solved.
 *
 * \param   V is a reference the vector field denoting the velocity field
 * \param   P is a reference the scalar field denoting the pressure field
 * \param   gamma is a constant of RK3 sub-step
 * \param   zeta is a constant of RK3 sub-step
 ********************************************************************************************************************************************
 */
void wallModel::advanceEta0(vfield &V, sfield &P, real gamma, real zeta) {
    //std::cout << V.Vx.F.lbound() << V.Vx.F.ubound() << V.Vx.F(-2, -2, 0) << std::endl;
    //std::cout << wallNum << "\t" << shiftVal << "\t" << V.Vx.F(5, 5, dlBnd(2)) << "\t" << V.Vx.F(5, 5, dlBnd(2) + shiftVal) << std::endl;
    //std::cout << wallNum << "\t" << shiftVal << "\t" << V.Vx.F(5, 6, dlBnd(2)) << "\t" << V.Vx.F(5, 6, dlBnd(2) + shiftVal) << std::endl;
    //std::cout << wallNum << "\t" << shiftVal << "\t" << V.Vx.F(6, 5, dlBnd(2)) << "\t" << V.Vx.F(6, 5, dlBnd(2) + shiftVal) << std::endl;
    //std::cout << wallNum << "\t" << shiftVal << "\t" << V.Vx.F(6, 6, dlBnd(2)) << "\t" << V.Vx.F(6, 6, dlBnd(2) + shiftVal) << std::endl;
    real hi = mesh.dXi;
    real hj = mesh.dEt;

    // Following indexing is applicable only for top and bottom walls (walls perpendicular to z direction)
    for (int i=dlBnd(0); i<=duBnd(0); i++) {
        for (int j=dlBnd(1); j<=duBnd(1); j++) {
            real dp_di, dp_dj;
            real nlTerm, prTerm;
            real dvii_di, dvij_di, dvij_dj, dvjj_dj;

            q(i, j) = sqrt(pow(vi(i, j), 2) + pow(vj(i, j), 2));

            dvii_di = mesh.xi_x(i)*(vii(i+1, j) - vii(i-1, j) + Tii(i+1, j) - Tii(i-1, j))/(2*hi);
            dvij_di = mesh.xi_x(i)*(vij(i+1, j) - vij(i-1, j) + Tij(i+1, j) - Tij(i-1, j))/(2*hi);
            dvij_dj = mesh.et_y(j)*(vij(i, j+1) - vij(i, j-1) + Tij(i, j+1) - Tij(i, j-1))/(2*hj);
            dvjj_dj = mesh.et_y(j)*(vjj(i, j+1) - vjj(i, j-1) + Tjj(i, j+1) - Tjj(i, j-1))/(2*hj);

            nlTerm = vi(i, j)*dvii_di + vi(i, j)*dvij_dj + vj(i, j)*dvij_di + vj(i, j)*dvjj_dj;

            dp_di = mesh.xi_x(i)*(P.F.F(i+1, j, wInd) - P.F.F(i-1, j, wInd))/(2*hi);
            dp_dj = mesh.et_y(j)*(P.F.F(i, j+1, wInd) - P.F.F(i, j-1, wInd))/(2*hj);

            prTerm = vi(i, j)*dp_di + vj(i, j)*dp_dj;
        }
    }
}


/**
 ********************************************************************************************************************************************
 * \brief   Function to calculate the Karman constant-like parameter
 *
 *          This function is called called by the SSV LES module at the boundaries
 *
 * \param   K is the subgrid energy in near wall cell
 * \param   Tik is one component of stress tensor in wall normal direction
 * \param   Tjk is second component of stress tensor in wall normal direction
 ********************************************************************************************************************************************
 */
real wallModel::updateK0(real K, real Tik, real Tjk) {
    return 0.45*sqrt(K)/(2*sqrt(sqrt(pow(Tik, 2) + pow(Tjk, 2))));
}


/**
 ********************************************************************************************************************************************
 * \brief   Function to compute slip velocity at wall from eta0
 *
 *          This function is called for each point on the modelled wall
 *
 * \param   uTau is the friction velocity computed from eta0
 * \param   dynKarm is the dynamically computed Karman constant-like parameter
 ********************************************************************************************************************************************
 */
void wallModel::computeBCVel() {
    real uMag, uTau;

    for (int i=dlBnd(0); i<=duBnd(0); i++) {
        for (int j=dlBnd(1); j<=duBnd(1); j++) {
            for (int k=dlBnd(2); k<=duBnd(2); k++) {
                uTau = sqrt(nu*fabs(eta0(i, j, 0)));
                uMag = uTau2u(uTau, K0(i, j, 0));

                // Compute these values from uMag at wall
                bcU(i, j, 0) = uMag;
                bcV(i, j, 0) = 0.0;
                bcW(i, j, 0) = 0.0;
            }
        }
    }
}


/**
 ********************************************************************************************************************************************
 * \brief   Function to compute slip velocity at wall from friction velocity
 *
 *          This function is called for each point on the modelled wall
 *
 * \param   uTau is the friction velocity computed from eta0
 * \param   dynKarm is the dynamically computed Karman constant-like parameter
 ********************************************************************************************************************************************
 */
inline real wallModel::uTau2u(real uTau, real dynKarm) {
    real h0Plus = h0*uTau/nu;

    if (h0Plus > h_nu_plus)
        return uTau*(log(h0Plus/h_nu_plus)/dynKarm + h_nu_plus);
    else
        return uTau*h0Plus;
}

