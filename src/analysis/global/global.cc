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
/*! \file global.cc
 *
 *  \brief Definitions for global functions/variables for post-processing
 *  \sa unittest.h
 *  \author Roshan Samuel
 *  \date May 2022
 *  \copyright New BSD License
 *
 ********************************************************************************************************************************************
 */

#include "global.h"

global::global(const grid &mesh): mesh(mesh) {
    // Loop limits of arrays for mid-point method of integration
    xlMP = mesh.coreDomain.lbound(0);
    xuMP = mesh.coreDomain.ubound(0);

    ylMP = mesh.coreDomain.lbound(1);
    yuMP = mesh.coreDomain.ubound(1);

    zlMP = mesh.coreDomain.lbound(2);
    zuMP = mesh.coreDomain.ubound(2);

    xfr = (mesh.rankData.xRank == 0)? true: false;
    yfr = (mesh.rankData.yRank == 0)? true: false;
    zfr = (mesh.rankData.zRank == 0)? true: false;

    xlr = (mesh.rankData.xRank == mesh.rankData.npX - 1)? true: false;
    ylr = (mesh.rankData.yRank == mesh.rankData.npY - 1)? true: false;
    zlr = (mesh.rankData.zRank == mesh.rankData.npZ - 1)? true: false;

    setWallRectDomains();
}


void global::setWallRectDomains() {
    blitz::RectDomain<3> core, full, wall;
    blitz::TinyVector<int, 3> lb, ub;

    core = mesh.coreDomain;
    full = mesh.fullDomain;

    lb = full.lbound();         lb(0) = 0;
    ub = full.ubound();         ub(0) = 0;
    wall = blitz::RectDomain<3>(lb, ub);
    x0Lft = wall;       x0Lft.lbound()(0) -= 1;         x0Lft.ubound()(0) -= 1;
    x0Rgt = wall;

    lb = full.lbound();         lb(0) = core.ubound(0);
    ub = full.ubound();         ub(0) = core.ubound(0);
    wall = blitz::RectDomain<3>(lb, ub);
    x1Lft = wall;
    x1Rgt = wall;       x1Rgt.lbound()(0) += 1;         x1Rgt.ubound()(0) += 1;


    lb = full.lbound();         lb(1) = 0;
    ub = full.ubound();         ub(1) = 0;
    wall = blitz::RectDomain<3>(lb, ub);
    y0Lft = wall;       y0Lft.lbound()(1) -= 1;         y0Lft.ubound()(1) -= 1;
    y0Rgt = wall;

    lb = full.lbound();         lb(1) = core.ubound(1);
    ub = full.ubound();         ub(1) = core.ubound(1);
    wall = blitz::RectDomain<3>(lb, ub);
    y1Lft = wall;
    y1Rgt = wall;       y1Rgt.lbound()(1) += 1;         y1Rgt.ubound()(1) += 1;


    lb = full.lbound();         lb(2) = 0;
    ub = full.ubound();         ub(2) = 0;
    wall = blitz::RectDomain<3>(lb, ub);
    z0Lft = wall;       z0Lft.lbound()(2) -= 1;         z0Lft.ubound()(2) -= 1;
    z0Rgt = wall;

    lb = full.lbound();         lb(2) = core.ubound(2);
    ub = full.ubound();         ub(2) = core.ubound(2);
    wall = blitz::RectDomain<3>(lb, ub);
    z1Lft = wall;
    z1Rgt = wall;       z1Rgt.lbound()(2) += 1;         z1Rgt.ubound()(2) += 1;
}


blitz::Array<real, 3> global::shift2Wall(blitz::Array<real, 3> F) {
    blitz::Array<real, 3> retMat;

    retMat.resize(F.extent());
    retMat.reindexSelf(F.lbound());

    retMat = F;

    if (xfr) retMat(x0Lft) = 0.5*(F(x0Rgt) + F(x0Lft));
    if (xlr) retMat(x1Rgt) = 0.5*(F(x1Rgt) + F(x1Lft));

    if (yfr) retMat(y0Lft) = 0.5*(F(y0Rgt) + F(y0Lft));
    if (ylr) retMat(y1Rgt) = 0.5*(F(y1Rgt) + F(y1Lft));

    if (zfr) retMat(z0Lft) = 0.5*(F(z0Rgt) + F(z0Lft));
    if (zlr) retMat(z1Rgt) = 0.5*(F(z1Rgt) + F(z1Lft));

    return retMat;
}


void global::initVBCs(vfield &V) {
    if (mesh.inputParams.probType == 3) {
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

    if (mesh.inputParams.probType == 1) {
        // NO-SLIP BCS FOR LDC
        V.uBot = new dirichlet(mesh, V.Vx, 4, 0.0);
        V.uTop = new dirichlet(mesh, V.Vx, 5, 1.0);
    } else {
        // NO-SLIP BCS
        V.uBot = new dirichlet(mesh, V.Vx, 4, 0.0);
        V.uTop = new dirichlet(mesh, V.Vx, 5, 0.0);
    }

#ifndef PLANAR
    if (mesh.inputParams.probType == 3) {
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

    // NO-SLIP BCS
    V.vBot = new dirichlet(mesh, V.Vy, 4, 0.0);
    V.vTop = new dirichlet(mesh, V.Vy, 5, 0.0);
#endif

    if (mesh.inputParams.probType == 3) {
        // INFLOW AND OUTFLOW BCS
        V.wLft = new dirichlet(mesh, V.Vz, 0, 0.0);
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

    // NO-SLIP BCS
    V.wBot = new dirichlet(mesh, V.Vz, 4, 0.0);
    V.wTop = new dirichlet(mesh, V.Vz, 5, 0.0);
}


void global::initTBCs(sfield &T) {
    // ADIABATIC BC FOR RBC, SST AND RRBC
    if (mesh.inputParams.probType == 5 || mesh.inputParams.probType == 6 || mesh.inputParams.probType == 8) {
        T.tLft = new neumann(mesh, T.F, 0, 0.0);
        T.tRgt = new neumann(mesh, T.F, 1, 0.0);

    // CONDUCTING BC FOR VERTICAL CONVECTION
    } else if (mesh.inputParams.probType == 7) {
        T.tLft = new dirichlet(mesh, T.F, 0, 1.0);
        T.tRgt = new dirichlet(mesh, T.F, 1, 0.0);
    }

#ifndef PLANAR
    T.tFrn = new neumann(mesh, T.F, 2, 0.0);
    T.tBak = new neumann(mesh, T.F, 3, 0.0);
#endif

    // HOT PLATE AT BOTTOM AND COLD PLATE AT TOP FOR RBC AND RRBC
    if (mesh.inputParams.probType == 5 || mesh.inputParams.probType == 8) {
        T.tBot = new dirichlet(mesh, T.F, 4, 1.0);
        T.tTop = new dirichlet(mesh, T.F, 5, 0.0);

    // COLD PLATE AT BOTTOM AND HOT PLATE AT TOP FOR SST
    } else if (mesh.inputParams.probType == 6) {
        T.tBot = new dirichlet(mesh, T.F, 4, 0.0);
        T.tTop = new dirichlet(mesh, T.F, 5, 1.0);
    }
}


void global::checkPeriodic(const parser &inputParams, parallel &rankData) {
    // Disable periodic data transfer by setting neighbouring ranks of boundary sub-domains to NULL
    // Left and right walls
    if (not inputParams.xPer) {
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


real global::simpsonBase(blitz::Array<real, 3> F, blitz::Array<real, 1> Z, blitz::Array<real, 1> Y, blitz::Array<real, 1> X) {
    blitz::Array<real, 1> h, hsum, hmul, hdiv;

    int N = Z.extent(0);
    h.resize(N-1);
    std::cout << F(0, 0, 0) << std::endl;
    //F.reindexSelf(blitz::TinyVector<int, 3>(-1, -1, -1));
    //std::cout << F(0, 0, 0) << std::endl;
    //MPI_Finalize();
    //exit(0);

    std::cout << Z.lbound() << Z.ubound() << std::endl;
    h = Z(blitz::Range(0, N, 1)) - Z(blitz::Range(-1, N-1, 1));
    //std::cout << h << std::endl;
    return 0;
}


real global::simpsonRule(blitz::Array<real, 3> F, blitz::Array<real, 1> Z, blitz::Array<real, 1> Y, blitz::Array<real, 1> X) {
    return simpsonBase(F, Z, Y, X);
}


real global::simpsonBase(blitz::Array<real, 2> F, blitz::Array<real, 1> Y, blitz::Array<real, 1> X) {
    return 0;
}


real global::simpsonRule(blitz::Array<real, 2> F, blitz::Array<real, 1> Y, blitz::Array<real, 1> X) {
    return simpsonBase(F, Y, X);
}


real global::simpsonBase(blitz::Array<real, 1> F, blitz::Array<real, 1> X) {
    blitz::Array<real, 1> h, hsum, hmul, hdiv;

    //N = 
    //h = 
    return 0;
}


real global::simpsonRule(blitz::Array<real, 1> F, blitz::Array<real, 1> X) {
    int xL = F.lbound(0);
    int xU = F.ubound(0);

    real intVal = 0;
    for (int i=xL; i<=xU; i++) {
        intVal += 1;
    }

    return 0;
}


real global::volAvgMidPt(blitz::Array<real, 3> F) {
    real dVol;
    real globalVal;
    real localVal = 0.0;
    real totalVol = mesh.xLen * mesh.yLen * mesh.zLen;

    for (int iX = xlMP; iX <= xuMP; iX++) {
        for (int iY = ylMP; iY <= yuMP; iY++) {
            for (int iZ = zlMP; iZ <= zuMP; iZ++) {
                dVol = (mesh.dXi/mesh.xi_x(iX))*(mesh.dEt/mesh.et_y(iY))*(mesh.dZt/mesh.zt_z(iZ));
                localVal += F(iX, iY, iZ)*dVol;
            }
        }
    }

    MPI_Allreduce(&localVal, &globalVal, 1, MPI_FP_REAL, MPI_SUM, MPI_COMM_WORLD);
    globalVal /= totalVol;

    return globalVal;
}
