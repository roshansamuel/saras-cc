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
wallModel::wallModel(const grid &mesh, const int bcWall): wallNum(bcWall), mesh(mesh) {
    dSize = mesh.fullDomain.ubound() - mesh.fullDomain.lbound() + 1;
    dlBnd = mesh.fullDomain.lbound();
    duBnd = mesh.fullDomain.ubound();

    // By default, rankFlag is true. i.e., the BC will be applied on all sub-domains.
    // This has to be changed appropriately.
    rankFlag = true;

    // By default, shiftVal is 1. i.e., the BC will be applied correctly only on the left wall along a given direction.
    // This has to be changed appropriately for the wall on the other side.
    shiftVal = 1;

    // Update rankFlag for the left wall (along X)
    if (wallNum == 0) rankFlag = mesh.rankData.xRank == 0;
    // Update rankFlag and shiftVal for the right wall (along X)
    if (wallNum == 1) {
        rankFlag = mesh.rankData.xRank == mesh.rankData.npX - 1;
        shiftVal = -1;
    }

    // Update rankFlag for the front wall (along Y)
    if (wallNum == 2) rankFlag = mesh.rankData.yRank == 0;
    // Update rankFlag and shiftVal for the back wall (along Y)
    if (wallNum == 3) {
        rankFlag = mesh.rankData.yRank == mesh.rankData.npY - 1;
        shiftVal = -1;
    }

    // Update rankFlag for the bottom wall (along Z)
    if (wallNum == 4) rankFlag = mesh.rankData.zRank == 0;
    // Update rankFlag and shiftVal for the top wall (along Z)
    if (wallNum == 5) {
        rankFlag = mesh.rankData.zRank == mesh.rankData.npZ - 1;
        shiftVal = -1;
    }

    // Find the dimension perpendicular to which wall-model is being applied (X -> 0, Y -> 1, Z -> 2) using the wallNum
    shiftDim = (int) wallNum/2;

    // Set size and lbounds of the wall slice according to the value of shiftDim
    switch (shiftDim) {
        case 0:
            dSize(0) = 1;
            dlBnd(0) = duBnd(0) = (shiftVal > 0)? 0: mesh.coreDomain.ubound(0);

            break;
        case 1:
            dSize(1) = 1;
            dlBnd(1) = duBnd(1) = (shiftVal > 0)? 0: mesh.coreDomain.ubound(1);

            break;
        case 2:
            dSize(2) = 1;
            dlBnd(2) = duBnd(2) = (shiftVal > 0)? 0: mesh.coreDomain.ubound(2);

            break;
    }

    // Resize all arrays using the size and lbounds of wall slice
    q.resize(dSize);            q.reindexSelf(dlBnd);
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

    for (int i=dlBnd(0); i<=duBnd(0); i++) {
        for (int j=dlBnd(1); j<=duBnd(1); j++) {
            for (int k=dlBnd(2); k<=duBnd(2); k++) {
                /*
                real vi, vj, vii, vij, vjj;

                // Hard coded for Z-direction
                vi = V.Vx.F(i, j, k);
                vj = V.Vy.F(i, j, k);
                q(i, j, k) = sqrt(pow(vi, 2) + pow(vj, 2));
                */

                if (i==5 and j==5) std::cout << dlBnd << V.Vx.F(i, j, k) << "\t" << Tii(i, j, 0) << "\t" << vi(i, j, 0) << "\t" << i << j << k<< std::endl;
            }
        }
    }

    /*
    real sNorm, tNorm;
    real sQx, sQy, sQz;
    real nuTurb, sgDiss;
    real localSGKE, localDisp, localNuSG;

    V.syncAll();

    localSGKE = 0.0;
    for (int iX = xS; iX <= xE; iX++) {
        real dx = mesh.x(iX) - mesh.x(iX - 1);
        for (int iY = yS; iY <= yE; iY++) {
            real dy = mesh.y(iY) - mesh.y(iY - 1);
            for (int iZ = zS; iZ <= zE; iZ++) {
                real dz = mesh.z(iZ) - mesh.z(iZ - 1);

                // Specify the values of all the quantities necessary for sgsFlux function to
                // compute the sub-grid stress correctly. These are all member variables of the les
                // class and hence globally available to all the member functions of the class.

                // The required quantities are:
                // 1. Cutoff wavelength
                del = std::pow(dx*dy*dz, 1.0/3.0);

                // 2. Velocities at the 3 x 3 x 3 points over which structure function will be calculated
                u = V.Vx.F(blitz::Range(iX-1, iX+1), blitz::Range(iY-1, iY+1), blitz::Range(iZ-1, iZ+1));
                v = V.Vy.F(blitz::Range(iX-1, iX+1), blitz::Range(iY-1, iY+1), blitz::Range(iZ-1, iZ+1));
                w = V.Vz.F(blitz::Range(iX-1, iX+1), blitz::Range(iY-1, iY+1), blitz::Range(iZ-1, iZ+1));

                // 3. The x, y and z coordinates of the 3 x 3 x 3 points over which u, v and w have been specified
                x = mesh.x(blitz::Range(iX-1, iX+1));
                y = mesh.y(blitz::Range(iY-1, iY+1));
                z = mesh.z(blitz::Range(iZ-1, iZ+1));

                // 4. The velocity gradient tensor specified as a 3 x 3 matrix
                dudx = A11(iX, iY, iZ), A12(iX, iY, iZ), A13(iX, iY, iZ),
                       A21(iX, iY, iZ), A22(iX, iY, iZ), A23(iX, iY, iZ),
                       A31(iX, iY, iZ), A32(iX, iY, iZ), A33(iX, iY, iZ);

                // Now the sub-grid stress can be calculated
                sgsStress(&sTxx, &sTyy, &sTzz, &sTxy, &sTyz, &sTzx);

                // Copy the calculated values to the sub-grid stress tensor field
                Txx->F.F(iX, iY, iZ) = sTxx;
                Tyy->F.F(iX, iY, iZ) = sTyy;
                Tzz->F.F(iX, iY, iZ) = sTzz;
                Txy->F.F(iX, iY, iZ) = sTxy;
                Tyz->F.F(iX, iY, iZ) = sTyz;
                Tzx->F.F(iX, iY, iZ) = sTzx;

                if (sgfFlag) {
                    // 5. The temperature gradient vector specified as a 3 component tiny vector
                    // To compute sub-grid scalar flux, the sgsStress calculations have already provided
                    // most of the necessary values. Only an additional temperature gradient vector is needed
                    dsdx = B1(iX, iY, iZ), B2(iX, iY, iZ), B3(iX, iY, iZ);

                    // Now the sub-grid scalar flus can be calculated
                    sgsFlux(&sQx, &sQy, &sQz);

                    // Copy the calculated values to the sub-grid scalar flux vector field
                    qX->F.F(iX, iY, iZ) = sQx;
                    qY->F.F(iX, iY, iZ) = sQy;
                    qZ->F.F(iX, iY, iZ) = sQz;
                }

                // Subgrid kinetic energy
                localSGKE += abs(K)*(mesh.dXi/mesh.xi_x(iX))*(mesh.dEt/mesh.et_y(iY))*(mesh.dZt/mesh.zt_z(iZ));

                // Subgrid dissipation
                sgDiss = -(Sxx*sTxx + Syy*sTyy + Szz*sTzz + 2.0*(Sxy*sTxy + Syz*sTyz + Szx*sTzx));
                localDisp += sgDiss*(mesh.dXi/mesh.xi_x(iX))*(mesh.dEt/mesh.et_y(iY))*(mesh.dZt/mesh.zt_z(iZ));

                // This below line hides an utterly perplexing Heisenbug which has led me to my wits end.
                // For some rare cases (with very low LES contribution), localDisp becomes NaN.
                // However, if you try to print the indices corresponding to the occurence of NaN, it disappears.
                if (std::isnan(localDisp)) localDisp = 0;

                // Compute turbulent viscosity norms of T_ij and S_ij tensors
                sNorm = std::sqrt(Sxx*Sxx + Syy*Syy + Szz*Szz + 2.0*(Sxy*Sxy + Syz*Syz + Szx*Szx));
                tNorm = std::sqrt(sTxx*sTxx + sTyy*sTyy + sTzz*sTzz + 2.0*(sTxy*sTxy + sTyz*sTyz + sTzx*sTzx));
                nuTurb = (tNorm/sNorm)/2;
                localNuSG += nuTurb*(mesh.dXi/mesh.xi_x(iX))*(mesh.dEt/mesh.et_y(iY))*(mesh.dZt/mesh.zt_z(iZ));
            }
        }
    }

    MPI_Allreduce(&localSGKE, &totalSGKE, 1, MPI_FP_REAL, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&localDisp, &totalDisp, 1, MPI_FP_REAL, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&localNuSG, &totalNuSG, 1, MPI_FP_REAL, MPI_SUM, MPI_COMM_WORLD);

    // Synchronize the sub-grid stress tensor field data across MPI processors
    Txx->syncFaces();
    Tyy->syncFaces();
    Tzz->syncFaces();
    Txy->syncFaces();
    Tyz->syncFaces();
    Tzx->syncFaces();

    // Compute the components of the divergence of sub-grid stress tensor field
    Txx->derS.calcDerivative1_x(A11);
    Txy->derS.calcDerivative1_x(A12);
    Tzx->derS.calcDerivative1_x(A13);
    Txy->derS.calcDerivative1_y(A21);
    Tyy->derS.calcDerivative1_y(A22);
    Tyz->derS.calcDerivative1_y(A23);
    Tzx->derS.calcDerivative1_z(A31);
    Tyz->derS.calcDerivative1_z(A32);
    Tzz->derS.calcDerivative1_z(A33);

    // Compute the divergence of the sub-grid stress tensor field
    B1 = A11 + A21 + A31;
    B2 = A12 + A22 + A32;
    B3 = A13 + A23 + A33;

    // Subtract the divergence to the RHS of NSE provided as argument to the function
    nseRHS.Vx(core) = nseRHS.Vx(core) - B1(core);
    nseRHS.Vy(core) = nseRHS.Vy(core) - B2(core);
    nseRHS.Vz(core) = nseRHS.Vz(core) - B3(core);

    if (sgfFlag) {
        // Synchronize the sub-grid scalar flux vector field data across MPI processors
        qX->syncFaces();
        qY->syncFaces();
        qZ->syncFaces();

        // Compute the components of the divergence of sub-grid scalar flux vector field
        qX->derS.calcDerivative1_x(B1);
        qY->derS.calcDerivative1_y(B2);
        qZ->derS.calcDerivative1_z(B3);

        // Sum the components to get the divergence of scalar flux, and subtract its contribution
        // to the RHS of the temperature field equation provided as argument to the function
        tmpRHS.F(core) = tmpRHS.F(core) - (B1(core) + B2(core) + B3(core));

    } else {
        V.derVx.calcDerivative2xx(A11);
        V.derVx.calcDerivative2yy(A12);
        V.derVx.calcDerivative2zz(A13);
        V.derVy.calcDerivative2xx(A21);
        V.derVy.calcDerivative2yy(A22);
        V.derVy.calcDerivative2zz(A23);
        V.derVz.calcDerivative2xx(A31);
        V.derVz.calcDerivative2yy(A32);
        V.derVz.calcDerivative2zz(A33);

        A11 = A11 + A12 + A13;
        A22 = A21 + A22 + A23;
        A33 = A31 + A32 + A33;

        B1 /= A11;
        B2 /= A22;
        B3 /= A33;

        B1 = (B1 + B2 + B3)/(mesh.inputParams.Pr*3.0);

        T.derS.calcDerivative2xx(A11);
        T.derS.calcDerivative2yy(A22);
        T.derS.calcDerivative2zz(A33);

        tmpRHS.F(core) = tmpRHS.F(core) - B1(core)*(A11(core) + A22(core) + A33(core));
    }
    */
}

