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
/*! \file vfield.cc
 *
 *  \brief Definitions for functions of class vfield - vector field
 *  \sa vfield.h
 *  \author Roshan Samuel, Ali Asad
 *  \date Nov 2019
 *  \copyright New BSD License
 *
 ********************************************************************************************************************************************
 */

#include "plainsf.h"
#include "plainvf.h"
#include "vfield.h"

/**
 ********************************************************************************************************************************************
 * \brief   Constructor of the vfield class
 *
 *          Three instances of field class are initialized.
 *          Each instance corresponds to a component of the vector field.
 *          The fields are appropriately staggered to place them on the cell faces.
 *          The vector field is also assigned a name.
 *
 * \param   gridData is a const reference to the global data in the grid class
 * \param   fieldName is a string value used to name and identify the vector field
 ********************************************************************************************************************************************
 */
vfield::vfield(const grid &gridData, std::string fieldName):
               gridData(gridData),
               Vx(gridData, "Vx"), Vy(gridData, "Vy"), Vz(gridData, "Vz"),
               derVx(gridData, Vx.F), derVy(gridData, Vy.F), derVz(gridData, Vz.F)
{
    this->fieldName = fieldName;

    derivTemp.resize(Vx.fSize);
    derivTemp.reindexSelf(Vx.flBound);

    core = gridData.coreDomain;

    // The following average grid spacings in physical plane are used when
    // calculating time-step using CFL condition in computeTStp(dt) function.
    avgDx = gridData.xLen/gridData.globalSize(0);
#ifndef PLANAR
    avgDy = gridData.yLen/gridData.globalSize(1);
#endif
    avgDz = gridData.zLen/gridData.globalSize(2);
}

/**
 ********************************************************************************************************************************************
 * \brief   Function to compute the diffusion term
 *
 *          The diffusion term (grad-squared) is caulculated here.
 *          The second derivatives of each component field are calculated along x, y and z.
 *          These terms are added to the corresponding components of the given plain
 *          vector field (plainvf), which is usually the RHS of the PDE being solved.
 *
 * \param   H is a reference to the plainvf into which the output is written
 ********************************************************************************************************************************************
 */
void vfield::computeDiff(plainvf &H) {
    derivTemp = 0.0;
    derVx.calcDerivative2xx(derivTemp);
    H.Vx(core) += derivTemp(core);

#ifndef PLANAR
    derivTemp = 0.0;
    derVx.calcDerivative2yy(derivTemp);
    H.Vx(core) += derivTemp(core);
#endif

    derivTemp = 0.0;
    derVx.calcDerivative2zz(derivTemp);
    H.Vx(core) += derivTemp(core);

#ifndef PLANAR
    derivTemp = 0.0;
    derVy.calcDerivative2xx(derivTemp);
    H.Vy(core) += derivTemp(core);

    derivTemp = 0.0;
    derVy.calcDerivative2yy(derivTemp);
    H.Vy(core) += derivTemp(core);

    derivTemp = 0.0;
    derVy.calcDerivative2zz(derivTemp);
    H.Vy(core) += derivTemp(core);
#endif

    derivTemp = 0.0;
    derVz.calcDerivative2xx(derivTemp);
    H.Vz(core) += derivTemp(core);

#ifndef PLANAR
    derivTemp = 0.0;
    derVz.calcDerivative2yy(derivTemp);
    H.Vz(core) += derivTemp(core);
#endif

    derivTemp = 0.0;
    derVz.calcDerivative2zz(derivTemp);
    H.Vz(core) += derivTemp(core);
}

/**
 ********************************************************************************************************************************************
 * \brief   Function to compute the convective derivative of the vector field
 *
 *          The function calculates \f$ (\mathbf{u}.\nabla)\mathbf{v} \f$ on the vector field, \f$\mathbf{v}\f$.
 *          To do so, the function needs the vector field (vfield) of velocity, \f$\mathbf{u}\f$.
 *
 * \param   V is a const reference to the vfield denoting convection velocity
 * \param   H is a reference to the plainvf into which the output is written
 ********************************************************************************************************************************************
 */
void vfield::computeNLin(const vfield &V, plainvf &H) {
    if (gridData.inputParams.upwindFlag) {
        upwindNLin(V, H);
    } else {
        derivTemp = 0.0;
        derVx.calcDerivative1_x(derivTemp);
        H.Vx(core) -= V.Vx.F(core)*derivTemp(core);

#ifndef PLANAR
        derivTemp = 0.0;
        derVx.calcDerivative1_y(derivTemp);
        H.Vx(core) -= V.Vy.F(core)*derivTemp(core);
#endif

        derivTemp = 0.0;    
        derVx.calcDerivative1_z(derivTemp);
        H.Vx(core) -= V.Vz.F(core)*derivTemp(core);

#ifndef PLANAR
        derivTemp = 0.0;
        derVy.calcDerivative1_x(derivTemp);
        H.Vy(core) -= V.Vx.F(core)*derivTemp(core);

        derivTemp = 0.0;
        derVy.calcDerivative1_y(derivTemp);
        H.Vy(core) -= V.Vy.F(core)*derivTemp(core);

        derivTemp = 0.0;
        derVy.calcDerivative1_z(derivTemp);
        H.Vy(core) -= V.Vz.F(core)*derivTemp(core);
#endif

        derivTemp = 0.0;
        derVz.calcDerivative1_x(derivTemp);
        H.Vz(core) -= V.Vx.F(core)*derivTemp(core);

#ifndef PLANAR
        derivTemp = 0.0;
        derVz.calcDerivative1_y(derivTemp);
        H.Vz(core) -= V.Vy.F(core)*derivTemp(core);
#endif

        derivTemp = 0.0;
        derVz.calcDerivative1_z(derivTemp);
        H.Vz(core) -= V.Vz.F(core)*derivTemp(core);
    }
}

/**
 ********************************************************************************************************************************************
 * \brief   Function to compute the convective derivative of the vector field with upwinding
 *
 *          The function calculates \f$ (\mathbf{u}.\nabla)\mathbf{v} \f$ on the vector field, \f$\mathbf{v}\f$.
 *          To do so, the function needs the vector field (vfield) of velocity, \f$\mathbf{u}\f$.
 *
 * \param   V is a const reference to the vfield denoting convection velocity
 * \param   H is a reference to the plainvf into which the output is written
 ********************************************************************************************************************************************
 */
void vfield::upwindNLin(const vfield &V, plainvf &H) {
    real pe;
    real u, dh, i2dh;
    real omega, a, b, c, d;
    bool xfr, xlr, yfr, ylr, zfr, zlr;

    xfr = xlr = yfr = ylr = zfr = zlr = false;

    if (gridData.rankData.xRank == 0) xfr = true;
    if (gridData.rankData.yRank == 0) yfr = true;
    if (gridData.rankData.zRank == 0) zfr = true;

    if (gridData.rankData.xRank == gridData.rankData.npX - 1) xlr = true;
    if (gridData.rankData.yRank == gridData.rankData.npY - 1) ylr = true;
    if (gridData.rankData.zRank == gridData.rankData.npZ - 1) zlr = true;

    // Parameter to adjust bias of upwinding
    // omega is the weight to the central difference stencil used in upwinding
    // Correspondingly the biasing stencil is weighted by (1.0 - omega)
    omega = gridData.inputParams.upParam;

    // The coefficients of the biased stencil are set using omega
    a = 1.0 - omega;
    b = 4.0 - 3.0*omega;
    c = 3.0*(1 - omega);
    d = -omega;

    for (int iX = 0; iX <= core.ubound(0); iX++) {
        for (int iY = 0; iY <= core.ubound(1); iY++) {
            for (int iZ = 0; iZ <= core.ubound(2); iZ++) {
                u = V.Vx.F(iX, iY, iZ);
                i2dh = 1.0/(2.0*gridData.dXi);
                if (((iX == 0) and xfr) or ((iX == core.ubound(0)) and xlr)) {
                    // Central difference for first and last point
                    H.Vx(iX, iY, iZ) -= u*gridData.xi_x(iX)*(Vx.F(iX+1, iY, iZ) - Vx.F(iX-1, iY, iZ))*i2dh;
                    H.Vy(iX, iY, iZ) -= u*gridData.xi_x(iX)*(Vy.F(iX+1, iY, iZ) - Vy.F(iX-1, iY, iZ))*i2dh;
                    H.Vz(iX, iY, iZ) -= u*gridData.xi_x(iX)*(Vz.F(iX+1, iY, iZ) - Vz.F(iX-1, iY, iZ))*i2dh;
                } else {
                    // First compute Peclet number
                    dh = gridData.x(iX+1) - gridData.x(iX);
                    pe = std::fabs(u)*dh/diffCoeff;

                    // If Peclet number is less than given limit, use central differencing, else biased stencils
                    if (pe < gridData.inputParams.peLimit) {
                        // Central difference
                        H.Vx(iX, iY, iZ) -= u*gridData.xi_x(iX)*(-Vx.F(iX+2, iY, iZ) + 8.0*Vx.F(iX+1, iY, iZ) - 8.0*Vx.F(iX-1, iY, iZ) + Vx.F(iX-2, iY, iZ))*i2dh/6.0;
                        H.Vy(iX, iY, iZ) -= u*gridData.xi_x(iX)*(-Vy.F(iX+2, iY, iZ) + 8.0*Vy.F(iX+1, iY, iZ) - 8.0*Vy.F(iX-1, iY, iZ) + Vy.F(iX-2, iY, iZ))*i2dh/6.0;
                        H.Vz(iX, iY, iZ) -= u*gridData.xi_x(iX)*(-Vz.F(iX+2, iY, iZ) + 8.0*Vz.F(iX+1, iY, iZ) - 8.0*Vz.F(iX-1, iY, iZ) + Vz.F(iX-2, iY, iZ))*i2dh/6.0;
                    } else {
                        // When using biased stencils, choose the biasing according to the local advection velocity
                        if (u > 0) {
                            // Backward difference
                            H.Vx(iX, iY, iZ) -= u*gridData.xi_x(iX)*(a*Vx.F(iX-2, iY, iZ) - b*Vx.F(iX-1, iY, iZ) + c*Vx.F(iX, iY, iZ) - d*Vx.F(iX+1, iY, iZ))*i2dh;
                            H.Vy(iX, iY, iZ) -= u*gridData.xi_x(iX)*(a*Vy.F(iX-2, iY, iZ) - b*Vy.F(iX-1, iY, iZ) + c*Vy.F(iX, iY, iZ) - d*Vy.F(iX+1, iY, iZ))*i2dh;
                            H.Vz(iX, iY, iZ) -= u*gridData.xi_x(iX)*(a*Vz.F(iX-2, iY, iZ) - b*Vz.F(iX-1, iY, iZ) + c*Vz.F(iX, iY, iZ) - d*Vz.F(iX+1, iY, iZ))*i2dh;
                        } else {
                            // Forward difference
                            H.Vx(iX, iY, iZ) -= u*gridData.xi_x(iX)*(-a*Vx.F(iX+2, iY, iZ) + b*Vx.F(iX+1, iY, iZ) - c*Vx.F(iX, iY, iZ) + d*Vx.F(iX-1, iY, iZ))*i2dh;
                            H.Vy(iX, iY, iZ) -= u*gridData.xi_x(iX)*(-a*Vy.F(iX+2, iY, iZ) + b*Vy.F(iX+1, iY, iZ) - c*Vy.F(iX, iY, iZ) + d*Vy.F(iX-1, iY, iZ))*i2dh;
                            H.Vz(iX, iY, iZ) -= u*gridData.xi_x(iX)*(-a*Vz.F(iX+2, iY, iZ) + b*Vz.F(iX+1, iY, iZ) - c*Vz.F(iX, iY, iZ) + d*Vz.F(iX-1, iY, iZ))*i2dh;
                        }
                    }
                }

                u = V.Vy.F(iX, iY, iZ);
                i2dh = 1.0/(2.0*gridData.dEt);
                if (((iY == 0) and yfr) or ((iY == core.ubound(1)) and ylr)) {
                    // Central difference for first and last point
                    H.Vx(iX, iY, iZ) -= u*gridData.et_y(iY)*(Vx.F(iX, iY+1, iZ) - Vx.F(iX, iY-1, iZ))*i2dh;
                    H.Vy(iX, iY, iZ) -= u*gridData.et_y(iY)*(Vy.F(iX, iY+1, iZ) - Vy.F(iX, iY-1, iZ))*i2dh;
                    H.Vz(iX, iY, iZ) -= u*gridData.et_y(iY)*(Vz.F(iX, iY+1, iZ) - Vz.F(iX, iY-1, iZ))*i2dh;
                } else {
                    // First compute Peclet number
                    dh = gridData.y(iY+1) - gridData.y(iY);
                    pe = std::fabs(u)*dh/diffCoeff;

                    // If Peclet number is less than given limit, use central differencing, else biased stencils
                    if (pe < gridData.inputParams.peLimit) {
                        // Central difference
                        H.Vx(iX, iY, iZ) -= u*gridData.et_y(iY)*(-Vx.F(iX, iY+2, iZ) + 8.0*Vx.F(iX, iY+1, iZ) - 8.0*Vx.F(iX, iY-1, iZ) + Vx.F(iX, iY-2, iZ))*i2dh/6.0;
                        H.Vy(iX, iY, iZ) -= u*gridData.et_y(iY)*(-Vy.F(iX, iY+2, iZ) + 8.0*Vy.F(iX, iY+1, iZ) - 8.0*Vy.F(iX, iY-1, iZ) + Vy.F(iX, iY-2, iZ))*i2dh/6.0;
                        H.Vz(iX, iY, iZ) -= u*gridData.et_y(iY)*(-Vz.F(iX, iY+2, iZ) + 8.0*Vz.F(iX, iY+1, iZ) - 8.0*Vz.F(iX, iY-1, iZ) + Vz.F(iX, iY-2, iZ))*i2dh/6.0;
                    } else {
                        // When using biased stencils, choose the biasing according to the local advection velocity
                        if (u > 0) {
                            // Backward difference
                            H.Vx(iX, iY, iZ) -= u*gridData.et_y(iY)*(a*Vx.F(iX, iY-2, iZ) - b*Vx.F(iX, iY-1, iZ) + c*Vx.F(iX, iY, iZ) - d*Vx.F(iX, iY+1, iZ))*i2dh;
                            H.Vy(iX, iY, iZ) -= u*gridData.et_y(iY)*(a*Vy.F(iX, iY-2, iZ) - b*Vy.F(iX, iY-1, iZ) + c*Vy.F(iX, iY, iZ) - d*Vy.F(iX, iY+1, iZ))*i2dh;
                            H.Vz(iX, iY, iZ) -= u*gridData.et_y(iY)*(a*Vz.F(iX, iY-2, iZ) - b*Vz.F(iX, iY-1, iZ) + c*Vz.F(iX, iY, iZ) - d*Vz.F(iX, iY+1, iZ))*i2dh;
                        } else {
                            // Forward difference
                            H.Vx(iX, iY, iZ) -= u*gridData.et_y(iY)*(-a*Vx.F(iX, iY+2, iZ) + b*Vx.F(iX, iY+1, iZ) - c*Vx.F(iX, iY, iZ) + d*Vx.F(iX, iY-1, iZ))*i2dh;
                            H.Vy(iX, iY, iZ) -= u*gridData.et_y(iY)*(-a*Vy.F(iX, iY+2, iZ) + b*Vy.F(iX, iY+1, iZ) - c*Vy.F(iX, iY, iZ) + d*Vy.F(iX, iY-1, iZ))*i2dh;
                            H.Vz(iX, iY, iZ) -= u*gridData.et_y(iY)*(-a*Vz.F(iX, iY+2, iZ) + b*Vz.F(iX, iY+1, iZ) - c*Vz.F(iX, iY, iZ) + d*Vz.F(iX, iY-1, iZ))*i2dh;
                        }
                    }
                }

                u = V.Vz.F(iX, iY, iZ);
                i2dh = 1.0/(2.0*gridData.dZt);
                if (((iZ == 0) and zfr) or ((iZ == core.ubound(2)) and zlr)) {
                    // Central difference for first and last point
                    H.Vx(iX, iY, iZ) -= u*gridData.zt_z(iZ)*(Vx.F(iX, iY, iZ+1) - Vx.F(iX, iY, iZ-1))*i2dh;
                    H.Vy(iX, iY, iZ) -= u*gridData.zt_z(iZ)*(Vy.F(iX, iY, iZ+1) - Vy.F(iX, iY, iZ-1))*i2dh;
                    H.Vz(iX, iY, iZ) -= u*gridData.zt_z(iZ)*(Vz.F(iX, iY, iZ+1) - Vz.F(iX, iY, iZ-1))*i2dh;
                } else {
                    // First compute Peclet number
                    dh = gridData.z(iZ+1) - gridData.z(iZ);
                    pe = std::fabs(u)*dh/diffCoeff;

                    // If Peclet number is less than given limit, use central differencing, else biased stencils
                    if (pe < gridData.inputParams.peLimit) {
                        // Central difference
                        H.Vx(iX, iY, iZ) -= u*gridData.zt_z(iZ)*(-Vx.F(iX, iY, iZ+2) + 8.0*Vx.F(iX, iY, iZ+1) - 8.0*Vx.F(iX, iY, iZ-1) + Vx.F(iX, iY, iZ-2))*i2dh/6.0;
                        H.Vy(iX, iY, iZ) -= u*gridData.zt_z(iZ)*(-Vy.F(iX, iY, iZ+2) + 8.0*Vy.F(iX, iY, iZ+1) - 8.0*Vy.F(iX, iY, iZ-1) + Vy.F(iX, iY, iZ-2))*i2dh/6.0;
                        H.Vz(iX, iY, iZ) -= u*gridData.zt_z(iZ)*(-Vz.F(iX, iY, iZ+2) + 8.0*Vz.F(iX, iY, iZ+1) - 8.0*Vz.F(iX, iY, iZ-1) + Vz.F(iX, iY, iZ-2))*i2dh/6.0;
                    } else {
                        // When using biased stencils, choose the biasing according to the local advection velocity
                        if (u > 0) {
                            // Backward difference
                            H.Vx(iX, iY, iZ) -= u*gridData.zt_z(iZ)*(a*Vx.F(iX, iY, iZ-2) - b*Vx.F(iX, iY, iZ-1) + c*Vx.F(iX, iY, iZ) - d*Vx.F(iX, iY, iZ+1))*i2dh;
                            H.Vy(iX, iY, iZ) -= u*gridData.zt_z(iZ)*(a*Vy.F(iX, iY, iZ-2) - b*Vy.F(iX, iY, iZ-1) + c*Vy.F(iX, iY, iZ) - d*Vy.F(iX, iY, iZ+1))*i2dh;
                            H.Vz(iX, iY, iZ) -= u*gridData.zt_z(iZ)*(a*Vz.F(iX, iY, iZ-2) - b*Vz.F(iX, iY, iZ-1) + c*Vz.F(iX, iY, iZ) - d*Vz.F(iX, iY, iZ+1))*i2dh;
                        } else {
                            // Forward difference
                            H.Vx(iX, iY, iZ) -= u*gridData.zt_z(iZ)*(-a*Vx.F(iX, iY, iZ+2) + b*Vx.F(iX, iY, iZ+1) - c*Vx.F(iX, iY, iZ) + d*Vx.F(iX, iY, iZ-1))*i2dh;
                            H.Vy(iX, iY, iZ) -= u*gridData.zt_z(iZ)*(-a*Vy.F(iX, iY, iZ+2) + b*Vy.F(iX, iY, iZ+1) - c*Vy.F(iX, iY, iZ) + d*Vy.F(iX, iY, iZ-1))*i2dh;
                            H.Vz(iX, iY, iZ) -= u*gridData.zt_z(iZ)*(-a*Vz.F(iX, iY, iZ+2) + b*Vz.F(iX, iY, iZ+1) - c*Vz.F(iX, iY, iZ) + d*Vz.F(iX, iY, iZ-1))*i2dh;
                        }
                    }
                }
            }
        }
    }
}

/**
 ********************************************************************************************************************************************
 * \brief   Operator is used to calculate time step #dt using CFL Condition
 *           
 *          When the parameters specify that time-step must be adaptively
 *          calculated using the Courant Number given in the parameters file,
 *          this function will provide the \f$ dt \f$ using the maximum values
 *          of the components of the vfield.
 *
 * \param   dt is a reference to the real value into which the calculated value of time-step is written
 *********************************************************************************************************************************************
 */
void vfield::computeTStp(real &dt) {
    real locMax, gloMax;

#ifdef PLANAR
    locMax = blitz::max((blitz::abs(Vx.F(core))/avgDx) +
                        (blitz::abs(Vz.F(core))/avgDz));
#else
    locMax = blitz::max((blitz::abs(Vx.F(core))/avgDx) +
                        (blitz::abs(Vy.F(core))/avgDy) +
                        (blitz::abs(Vz.F(core))/avgDz));
#endif

    MPI_Allreduce(&locMax, &gloMax, 1, MPI_FP_REAL, MPI_MAX, MPI_COMM_WORLD);

    dt = gridData.inputParams.courantNumber/gloMax;
}

/**
 ********************************************************************************************************************************************
 * \brief   Operator to compute the divergence of the vector field
 *
 *          The operator computes the divergence of a face-centered staggered vector field, and stores it into a cell centered
 *          scalar field as defined by the tensor operation:
 *          \f$ \nabla . \mathbf{v} = \frac{\partial \mathbf{v}}{\partial x} +
 *                                    \frac{\partial \mathbf{v}}{\partial y} +
 *                                    \frac{\partial \mathbf{v}}{\partial z} \f$.
 *
 * \param   divV is a reference to the plain scalar field (plainsf) into which the computed divergence is written.
 ********************************************************************************************************************************************
 */
void vfield::divergence(plainsf &divV) {
    divV = 0.0;

    derivTemp = 0.0;
    derVx.calcDerivative1_x(derivTemp);
    divV.F(core) += derivTemp(core);

#ifndef PLANAR
    derivTemp = 0.0;
    derVy.calcDerivative1_y(derivTemp);
    divV.F(core) += derivTemp(core);
#endif

    derivTemp = 0.0;
    derVz.calcDerivative1_z(derivTemp);
    divV.F(core) += derivTemp(core);
}

/**
 ********************************************************************************************************************************************
 * \brief   Function to impose the boundary conditions for the X-component of the vector field
 *
 *          The function first calls the \ref field#syncFaces "syncFaces" function of the Vx field to update the sub-domain pads.
 *          Then the boundary conditions are applied at the full domain boundaries by calling the imposeBC()
 *          of each boundary class object assigned to each wall.
 *          The order of imposing boundary conditions is - left, right, front, back, bottom and top boundaries.
 *          The corner values are not being imposed specifically and is thus dependent on the above order.
 *
 ********************************************************************************************************************************************
 */
void vfield::imposeVxBC() {
    Vx.syncFaces();

    if (not gridData.inputParams.xPer) {
        uLft->imposeBC();
        uRgt->imposeBC();
    }
#ifndef PLANAR
    if (not gridData.inputParams.yPer) {
        uFrn->imposeBC();
        uBak->imposeBC();
    }
#endif
    if (not gridData.inputParams.zPer) {
        uTop->imposeBC();
        uBot->imposeBC();
    }
}

/**
 ********************************************************************************************************************************************
 * \brief   Function to impose the boundary conditions for the Y-component of the vector field
 *
 *          The function first calls the \ref field#syncFaces "syncFaces" function of the Vy field to update the sub-domain pads.
 *          Then the boundary conditions are applied at the full domain boundaries by calling the imposeBC()
 *          of each boundary class object assigned to each wall.
 *          The order of imposing boundary conditions is - left, right, front, back, bottom and top boundaries.
 *          The corner values are not being imposed specifically and is thus dependent on the above order.
 *
 ********************************************************************************************************************************************
 */
void vfield::imposeVyBC() {
    Vy.syncFaces();

    if (not gridData.inputParams.xPer) {
        vLft->imposeBC();
        vRgt->imposeBC();
    }
#ifndef PLANAR
    if (not gridData.inputParams.yPer) {
        vFrn->imposeBC();
        vBak->imposeBC();
    }
#endif
    if (not gridData.inputParams.zPer) {
        vTop->imposeBC();
        vBot->imposeBC();
    }
}

/**
 ********************************************************************************************************************************************
 * \brief   Function to impose the boundary conditions for the Z-component of the vector field
 *
 *          The function first calls the \ref field#syncFaces "syncFaces" function of the Vz field to update the sub-domain pads.
 *          Then the boundary conditions are applied at the full domain boundaries by calling the imposeBC()
 *          of each boundary class object assigned to each wall.
 *          The order of imposing boundary conditions is - left, right, front, back, bottom and top boundaries.
 *          The corner values are not being imposed specifically and is thus dependent on the above order.
 *
 ********************************************************************************************************************************************
 */
void vfield::imposeVzBC() {
    Vz.syncFaces();

    if (not gridData.inputParams.xPer) {
        wLft->imposeBC();
        wRgt->imposeBC();
    }
#ifndef PLANAR
    if (not gridData.inputParams.yPer) {
        wFrn->imposeBC();
        wBak->imposeBC();
    }
#endif
    if (not gridData.inputParams.zPer) {
        wTop->imposeBC();
        wBot->imposeBC();
    }
}

/**
 ********************************************************************************************************************************************
 * \brief   Function to impose the boundary conditions for all the components of the vfield
 *
 *          The function merely calls \ref vfield#imposeVxBC "imposeVxBC", \ref vfield#imposeVyBC "imposeVyBC"
 *          and \ref vfield#imposeVzBC "imposeVzBC" functions together.
 *
 ********************************************************************************************************************************************
 */
void vfield::imposeBCs() {
    imposeVxBC();
#ifndef PLANAR
    imposeVyBC();
#endif
    imposeVzBC();
}

/**
 ********************************************************************************************************************************************
 * \brief   Function to set the diffusion coefficient of the vector field
 *
 *          The function sets the diffusion coefficient corresponding to the vector field.
 *          This constant is used when computing the Peclet number when upwinding is enabled.
 *          Trying to use upwinding without setting this constant may cause the code to crash.
 *
 ********************************************************************************************************************************************
 */
void vfield::setDiffCoeff(real dCoeff) {
    diffCoeff = dCoeff;
}

/**
 ********************************************************************************************************************************************
 * \brief   Overloaded operator to add a given plain vector field
 *
 *          The unary operator += adds a given plain vector field to the vfield and returns
 *          a pointer to itself.
 *
 * \param   a is a reference to the plainvf to be added to the member fields
 *
 * \return  A pointer to itself is returned by the vector field object to which the operator belongs
 ********************************************************************************************************************************************
 */
vfield& vfield::operator += (plainvf &a) {
    Vx.F += a.Vx;
    Vy.F += a.Vy;
    Vz.F += a.Vz;

    return *this;
}

/**
 ********************************************************************************************************************************************
 * \brief   Overloaded operator to subtract a given plain vector field
 *
 *          The unary operator -= subtracts a given plain vector field from the vfield and returns
 *          a pointer to itself.
 *
 * \param   a is a reference to the plainvf to be subtracted from the member fields
 *
 * \return  A pointer to itself is returned by the vector field object to which the operator belongs
 ********************************************************************************************************************************************
 */
vfield& vfield::operator -= (plainvf &a) {
    Vx.F -= a.Vx;
    Vy.F -= a.Vy;
    Vz.F -= a.Vz;

    return *this;
}

/**
 ********************************************************************************************************************************************
 * \brief   Overloaded operator to add a given vector field
 *
 *          The unary operator += adds a given vector field to the vfield and returns
 *          a pointer to itself.
 *
 * \param   a is a reference to the vfield to be added to the member fields
 *
 * \return  A pointer to itself is returned by the vector field object to which the operator belongs
 ********************************************************************************************************************************************
 */
vfield& vfield::operator += (vfield &a) {
    Vx.F += a.Vx.F;
    Vy.F += a.Vy.F;
    Vz.F += a.Vz.F;

    return *this;
}

/**
 ********************************************************************************************************************************************
 * \brief   Overloaded operator to subtract a given vector field
 *
 *          The unary operator -= subtracts a given vector field from the vfield and returns
 *          a pointer to itself.
 *
 * \param   a is a reference to the vfield to be subtracted from the member fields
 *
 * \return  A pointer to itself is returned by the vector field object to which the operator belongs
 ********************************************************************************************************************************************
 */
vfield& vfield::operator -= (vfield &a) {
    Vx.F -= a.Vx.F;
    Vy.F -= a.Vy.F;
    Vz.F -= a.Vz.F;

    return *this;
}

/**
 ********************************************************************************************************************************************
 * \brief   Overloaded operator to multiply a scalar value to the vector field
 *
 *          The unary operator *= multiplies a real value to all the fields (Vx, Vy and Vz) stored in vfield and returns
 *          a pointer to itself.
 *
 * \param   a is a real number to be multiplied to the vector field
 *
 * \return  A pointer to itself is returned by the vector field class to which the operator belongs
 ********************************************************************************************************************************************
 */
vfield& vfield::operator *= (real a) {
    Vx.F *= a;
    Vy.F *= a;
    Vz.F *= a;

    return *this;
}

/**
 ********************************************************************************************************************************************
 * \brief   Overloaded operator to assign a plain vector field to the vector field
 *
 *          The operator = assigns all the three blitz arrays of a plain vector field (plainvf)
 *          to the corresponding arrays in the three fields of the vfield.
 *
 * \param   a is a plainvf to be assigned to the vector field
 ********************************************************************************************************************************************
 */
void vfield::operator = (plainvf &a) {
    Vx.F = a.Vx;
    Vy.F = a.Vy;
    Vz.F = a.Vz;
}

/**
 ********************************************************************************************************************************************
 * \brief   Overloaded operator to assign another vector field to the vector field
 *
 *          The operator = assigns all the three fields of a given vector field (vfield)
 *          to the corresponding fields of the vfield.
 *
 * \param   a is a vfield to be assigned to the vector field
 ********************************************************************************************************************************************
 */
void vfield::operator = (vfield &a) {
    Vx.F = a.Vx.F;
    Vy.F = a.Vy.F;
    Vz.F = a.Vz.F;
}

/**
 ********************************************************************************************************************************************
 * \brief   Overloaded operator to assign a scalar value to the vector field
 *
 *          The operator = assigns a real value to all the fields (Vx, Vy and Vz) stored in vfield.
 *
 * \param   a is a real number to be assigned to the vector field
 ********************************************************************************************************************************************
 */
void vfield::operator = (real a) {
    Vx.F = a;
    Vy.F = a;
    Vz.F = a;
}
