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

void initVBCs(const grid &mesh, vfield &V) {
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


void initTBCs(const grid &mesh, sfield &T) {
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


real simpson(blitz::Array<real, 3> F, blitz::Array<real, 1> Z, blitz::Array<real, 1> Y, blitz::Array<real, 1> X) {
    return 0;
}


real simpson(blitz::Array<real, 2> F, blitz::Array<real, 1> Y, blitz::Array<real, 1> X) {
    return 0;
}


real simpson(blitz::Array<real, 1> F, blitz::Array<real, 1> X) {
    int xL = F.lbound(0);
    int xU = F.ubound(0);

    real intVal = 0;
    for (int i=xL; i<=xU; i++) {
        intVal += 1;
    }

    return 0;
}


real volAvg(const grid &mesh, blitz::Array<real, 3> F) {
    real dVol;
    real globalVal;
    real localVal = 0.0;
    real totalVol = mesh.xLen * mesh.yLen * mesh.zLen;

    for (int iX = mesh.coreDomain.lbound(0); iX <= mesh.coreDomain.ubound(0); iX++) {
        for (int iY = mesh.coreDomain.lbound(1); iY <= mesh.coreDomain.ubound(1); iY++) {
            for (int iZ = mesh.coreDomain.lbound(2); iZ <= mesh.coreDomain.ubound(2); iZ++) {
                dVol = (mesh.dXi/mesh.xi_x(iX))*(mesh.dEt/mesh.et_y(iY))*(mesh.dZt/mesh.zt_z(iZ));
                localVal += F(iX, iY, iZ)*dVol;
            }
        }
    }

    MPI_Allreduce(&localVal, &globalVal, 1, MPI_FP_REAL, MPI_SUM, MPI_COMM_WORLD);
    globalVal /= totalVol;

    return globalVal;
}
