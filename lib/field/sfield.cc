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
/*! \file sfield.cc
 *
 *  \brief Definitions for functions of class sfield - scalar field
 *  \sa sfield.h
 *  \author Roshan Samuel, Ali Asad
 *  \date Nov 2019
 *  \copyright New BSD License
 *
 ********************************************************************************************************************************************
 */

#include "plainsf.h"
#include "plainvf.h"
#include "sfield.h"

/**
 ********************************************************************************************************************************************
 * \brief   Constructor of the sfield class
 *
 *          One instance of the field class to store the data of the scalar field is initialized.
 *          The field is initialized with appropriate grid staggering to place the scalar on the cell centres.
 *          The name for the vector field as given by the user is also assigned.
 *
 * \param   gridData is a const reference to the global data contained in the grid class
 * \param   fieldName is a string value set by the user to name and identify the scalar field
 ********************************************************************************************************************************************
 */
sfield::sfield(const grid &gridData, std::string fieldName):
               gridData(gridData),
               F(gridData, fieldName),
               derS(gridData, F.F)
{
    this->fieldName = fieldName;

    tempX.resize(F.fSize);
    tempX.reindexSelf(F.flBound);

#ifndef PLANAR
    tempY.resize(F.fSize);
    tempY.reindexSelf(F.flBound);
#endif

    tempZ.resize(F.fSize);
    tempZ.reindexSelf(F.flBound);

    core = gridData.coreDomain;

    xfr = xlr = yfr = ylr = zfr = zlr = false;

    if (gridData.rankData.xRank == 0) xfr = true;
    if (gridData.rankData.yRank == 0) yfr = true;
    if (gridData.rankData.zRank == 0) zfr = true;

    if (gridData.rankData.xRank == gridData.rankData.npX - 1) xlr = true;
    if (gridData.rankData.yRank == gridData.rankData.npY - 1) ylr = true;
    if (gridData.rankData.zRank == gridData.rankData.npZ - 1) zlr = true;

    firstOrder = false;
    // No, not the fascist regime - first order upwinding
    // By default the solver will try to use second order upwinding
    // If order of differentiation is set to 1 in parameters, it will
    // use first order upwinding instead.
    if (gridData.inputParams.dScheme == 1) firstOrder = true;

    // Parameter to adjust bias of upwinding
    // omega is the weight to the central difference stencil used in upwinding
    // Correspondingly the biasing stencil is weighted by (1.0 - omega)
    omega = gridData.inputParams.upParam;

    // The coefficients of the biased stencil are set using omega
    if (firstOrder) {
        // First order upwinding coefficients
        a = 2.0 - omega;
        b = 2.0*(1.0 - omega);
        c = -omega;
    } else {
        // Second order upwinding coefficients
        a = 1.0 - omega;
        b = 4.0 - 3.0*omega;
        c = 3.0*(1.0 - omega);
        d = -omega;
    }

    // The coefficients for finite-difference stencils
    i2dx = 1.0/(2.0*gridData.dXi);
    i2dy = 1.0/(2.0*gridData.dEt);
    i2dz = 1.0/(2.0*gridData.dZt);
}

/**
 ********************************************************************************************************************************************
 * \brief   Function to compute the diffusion term
 *
 *          The diffusion term (grad-squared) is caulculated here.
 *          The second derivatives of the scalar field are calculated along x, y and z.
 *          These terms are added to the given plain scalar field (plainsf),
 *          which is usually the RHS of the PDE being solved.
 *
 * \param   H is a reference to the plain scalar field (plainsf) to which the output will be written
 ********************************************************************************************************************************************
 */
void sfield::computeDiff(plainsf &H) {
    tempX = 0.0;
    derS.calcDerivative2xx(tempX);
    H.F(core) += tempX(core);

#ifndef PLANAR
    tempY = 0.0;
    derS.calcDerivative2yy(tempY);
    H.F(core) += tempY(core);
#endif

    tempZ = 0.0;
    derS.calcDerivative2zz(tempZ);
    H.F(core) += tempZ(core);
}

/**
 ********************************************************************************************************************************************
 * \brief   Function to compute the convective derivative of the scalar field
 *
 *          The function calculates \f$ (\mathbf{u}.\nabla)f \f$ at the grid nodes of the scalar field f.
 *          To do so, the function needs the vector field (vfield) of velocity, \f$\mathbf{u}\f$.
 *
 * \param   V is a const reference to a vector field (vfield) that specifies the convection velocity
 * \param   H is a reference to the plainsf into which the output is written
 ********************************************************************************************************************************************
 */
void sfield::computeNLin(const vfield &V, plainsf &H) {
    switch (gridData.inputParams.nlScheme) {
        case 1:
            tempX = 0.0;
            derS.calcDerivative1_x(tempX);
            H.F(core) -= V.Vx.F(core)*tempX(core);
#ifndef PLANAR
            tempY = 0.0;
            derS.calcDerivative1_y(tempY);
            H.F(core) -= V.Vy.F(core)*tempY(core);
#endif
            tempZ = 0.0;
            derS.calcDerivative1_z(tempZ);
            H.F(core) -= V.Vz.F(core)*tempZ(core);
            break;
        case 2:
            upwindNLin(V, H);
            break;
        case 3:
            morinishiNLin(V, H);
            break;
        default:
            if (gridData.pf) std::cout << "ERROR: Invalid parameter for non-linear computation scheme. ABORTING" << std::endl;
            MPI_Finalize();
            exit(0);
    }
}

/**
 ********************************************************************************************************************************************
 * \brief   Function to compute the non-linear term using Morinishi scheme
 *
 *          The function calculates the non-linear term at the grid nodes of the scalar field f.
 *          To do so, the function needs the vector field (vfield) of velocity, \f$\mathbf{u}\f$.
 *          The function splits the non-linear term into two parts as described in Morinishi's paper.
 *
 * \param   V is a const reference to the vector field (vfield) denoting convection velocity
 * \param   H is a reference to the plainsf into which the output is written
 ********************************************************************************************************************************************
 */
void sfield::morinishiNLin(const vfield &V, plainsf &H) {
    real advTerm, divTerm;

    tempX = V.Vx.F * F.F;
#ifndef PLANAR
    tempY = V.Vy.F * F.F;
#endif
    tempZ = V.Vz.F * F.F;

#ifdef PLANAR
    int iY = 0;
    for (int iX = 0; iX <= core.ubound(0); iX++) {
        for (int iZ = 0; iZ <= core.ubound(2); iZ++) {
            if (firstOrder) {
                // Second order central difference everywhere
                advTerm = V.Vx.F(iX, iY, iZ)*gridData.xi_x(iX)*(F.F(iX+1, iY, iZ) - F.F(iX-1, iY, iZ))*i2dx;
                divTerm = gridData.xi_x(iX)*(tempX(iX+1, iY, iZ) - tempX(iX-1, iY, iZ))*i2dx;
            } else {
                if (((iX == 0) and xfr) or ((iX == core.ubound(0)) and xlr)) {
                    // Second order central difference for first and last point
                    advTerm = V.Vx.F(iX, iY, iZ)*gridData.xi_x(iX)*(F.F(iX+1, iY, iZ) - F.F(iX-1, iY, iZ))*i2dx;
                    divTerm = gridData.xi_x(iX)*(tempX(iX+1, iY, iZ) - tempX(iX-1, iY, iZ))*i2dx;
                } else {
                    // Fourth order central difference elsewhere
                    advTerm = V.Vx.F(iX, iY, iZ)*gridData.xi_x(iX)*(-F.F(iX+2, iY, iZ) + 8.0*F.F(iX+1, iY, iZ) - 8.0*F.F(iX-1, iY, iZ) + F.F(iX-2, iY, iZ))*i2dx/6.0;
                    divTerm = gridData.xi_x(iX)*(-tempX(iX+2, iY, iZ) + 8.0*tempX(iX+1, iY, iZ) - 8.0*tempX(iX-1, iY, iZ) + tempX(iX-2, iY, iZ))*i2dx/6.0;
                }
            }

            H.F(iX, iY, iZ) -= (advTerm + divTerm)/2;

            if (firstOrder) {
                // Second order central difference everywhere
                advTerm = V.Vz.F(iX, iY, iZ)*gridData.zt_z(iZ)*(F.F(iX, iY, iZ+1) - F.F(iX, iY, iZ-1))*i2dz;
                divTerm = gridData.zt_z(iZ)*(tempZ(iX, iY, iZ+1) - tempZ(iX, iY, iZ-1))*i2dz;
            } else {
                if (((iZ == 0) and zfr) or ((iZ == core.ubound(1)) and zlr)) {
                    // Second order central difference for first and last point
                    advTerm = V.Vz.F(iX, iY, iZ)*gridData.zt_z(iZ)*(F.F(iX, iY, iZ+1) - F.F(iX, iY, iZ-1))*i2dz;
                    divTerm = gridData.zt_z(iZ)*(tempZ(iX, iY, iZ+1) - tempZ(iX, iY, iZ-1))*i2dz;
                } else {
                    // Fourth order central difference elsewhere
                    advTerm = V.Vz.F(iX, iY, iZ)*gridData.zt_z(iZ)*(-F.F(iX, iY, iZ+2) + 8.0*F.F(iX, iY, iZ+1) - 8.0*F.F(iX, iY, iZ-1) + F.F(iX, iY, iZ-2))*i2dz/6.0;
                    divTerm = gridData.zt_z(iZ)*(-tempZ(iX, iY, iZ+2) + 8.0*tempZ(iX, iY, iZ+1) - 8.0*tempZ(iX, iY, iZ-1) + tempZ(iX, iY, iZ-2))*i2dz/6.0;
                }
            }

            H.F(iX, iY, iZ) -= (advTerm + divTerm)/2;
        }
    }
#else
    for (int iX = 0; iX <= core.ubound(0); iX++) {
        for (int iY = 0; iY <= core.ubound(1); iY++) {
            for (int iZ = 0; iZ <= core.ubound(2); iZ++) {
                if (firstOrder) {
                    // Second order central difference everywhere
                    advTerm = V.Vx.F(iX, iY, iZ)*gridData.xi_x(iX)*(F.F(iX+1, iY, iZ) - F.F(iX-1, iY, iZ))*i2dx;
                    divTerm = gridData.xi_x(iX)*(tempX(iX+1, iY, iZ) - tempX(iX-1, iY, iZ))*i2dx;
                } else {
                    if (((iX == 0) and xfr) or ((iX == core.ubound(0)) and xlr)) {
                        // Second order central difference for first and last point
                        advTerm = V.Vx.F(iX, iY, iZ)*gridData.xi_x(iX)*(F.F(iX+1, iY, iZ) - F.F(iX-1, iY, iZ))*i2dx;
                        divTerm = gridData.xi_x(iX)*(tempX(iX+1, iY, iZ) - tempX(iX-1, iY, iZ))*i2dx;
                    } else {
                        // Fourth order central difference elsewhere
                        advTerm = V.Vx.F(iX, iY, iZ)*gridData.xi_x(iX)*(-F.F(iX+2, iY, iZ) + 8.0*F.F(iX+1, iY, iZ) - 8.0*F.F(iX-1, iY, iZ) + F.F(iX-2, iY, iZ))*i2dx/6.0;
                        divTerm = gridData.xi_x(iX)*(-tempX(iX+2, iY, iZ) + 8.0*tempX(iX+1, iY, iZ) - 8.0*tempX(iX-1, iY, iZ) + tempX(iX-2, iY, iZ))*i2dx/6.0;
                    }
                }

                H.F(iX, iY, iZ) -= (advTerm + divTerm)/2;

                if (firstOrder) {
                    // Second order central difference everywhere
                    advTerm = V.Vy.F(iX, iY, iZ)*gridData.et_y(iY)*(F.F(iX, iY+1, iZ) - F.F(iX, iY-1, iZ))*i2dy;
                    divTerm = gridData.et_y(iY)*(tempY(iX, iY+1, iZ) - tempY(iX, iY-1, iZ))*i2dy;
                } else {
                    if (((iY == 0) and yfr) or ((iY == core.ubound(1)) and ylr)) {
                        // Second order central difference for first and last point
                        advTerm = V.Vy.F(iX, iY, iZ)*gridData.et_y(iY)*(F.F(iX, iY+1, iZ) - F.F(iX, iY-1, iZ))*i2dy;
                        divTerm = gridData.et_y(iY)*(tempY(iX, iY+1, iZ) - tempY(iX, iY-1, iZ))*i2dy;
                    } else {
                        // Fourth order central difference elsewhere
                        advTerm = V.Vy.F(iX, iY, iZ)*gridData.et_y(iY)*(-F.F(iX, iY+2, iZ) + 8.0*F.F(iX, iY+1, iZ) - 8.0*F.F(iX, iY-1, iZ) + F.F(iX, iY-2, iZ))*i2dy/6.0;
                        divTerm = gridData.et_y(iY)*(-tempY(iX, iY+2, iZ) + 8.0*tempY(iX, iY+1, iZ) - 8.0*tempY(iX, iY-1, iZ) + tempY(iX, iY-2, iZ))*i2dy/6.0;
                    }
                }

                H.F(iX, iY, iZ) -= (advTerm + divTerm)/2;

                if (firstOrder) {
                    // Second order central difference everywhere
                    advTerm = V.Vz.F(iX, iY, iZ)*gridData.zt_z(iZ)*(F.F(iX, iY, iZ+1) - F.F(iX, iY, iZ-1))*i2dz;
                    divTerm = gridData.zt_z(iZ)*(tempZ(iX, iY, iZ+1) - tempZ(iX, iY, iZ-1))*i2dz;
                } else {
                    if (((iZ == 0) and zfr) or ((iZ == core.ubound(1)) and zlr)) {
                        // Second order central difference for first and last point
                        advTerm = V.Vz.F(iX, iY, iZ)*gridData.zt_z(iZ)*(F.F(iX, iY, iZ+1) - F.F(iX, iY, iZ-1))*i2dz;
                        divTerm = gridData.zt_z(iZ)*(tempZ(iX, iY, iZ+1) - tempZ(iX, iY, iZ-1))*i2dz;
                    } else {
                        // Fourth order central difference elsewhere
                        advTerm = V.Vz.F(iX, iY, iZ)*gridData.zt_z(iZ)*(-F.F(iX, iY, iZ+2) + 8.0*F.F(iX, iY, iZ+1) - 8.0*F.F(iX, iY, iZ-1) + F.F(iX, iY, iZ-2))*i2dz/6.0;
                        divTerm = gridData.zt_z(iZ)*(-tempZ(iX, iY, iZ+2) + 8.0*tempZ(iX, iY, iZ+1) - 8.0*tempZ(iX, iY, iZ-1) + tempZ(iX, iY, iZ-2))*i2dz/6.0;
                    }
                }

                H.F(iX, iY, iZ) -= (advTerm + divTerm)/2;
            }
        }
    }
#endif
}

/**
 ********************************************************************************************************************************************
 * \brief   Function to compute the convective derivative of the scalar field with upwinding
 *
 *          The function calculates \f$ (\mathbf{u}.\nabla)f \f$ at the grid nodes of the scalar field f.
 *          To do so, the function needs the vector field (vfield) of velocity, \f$\mathbf{u}\f$.
 *
 * \param   V is a const reference to a vector field (vfield) that specifies the convection velocity
 * \param   H is a reference to the plainsf into which the output is written
 ********************************************************************************************************************************************
 */
void sfield::upwindNLin(const vfield &V, plainsf &H) {
    real pe;
    real u, dh;

    if (firstOrder) {
        // FIRST ORDER UPWINDING
        // This uses first order forward and backward finite-difference stencils
        // mixed with second order central difference stencils where Pe is high.
        // Elsewhere it defaults to second order central-difference.
        for (int iX = 0; iX <= core.ubound(0); iX++) {
            for (int iY = 0; iY <= core.ubound(1); iY++) {
                for (int iZ = 0; iZ <= core.ubound(2); iZ++) {
                    u = V.Vx.F(iX, iY, iZ);
                    // First compute Peclet number
                    dh = gridData.x(iX+1) - gridData.x(iX);
                    pe = std::fabs(u)*dh/diffCoeff;

                    // If Peclet number is less than given limit, use central differencing, else biased stencils
                    if (pe < gridData.inputParams.peLimit) {
                        // Central difference
                        H.F(iX, iY, iZ) -= u*gridData.xi_x(iX)*(F.F(iX+1, iY, iZ) - F.F(iX-1, iY, iZ))*i2dx;
                    } else {
                        // When using biased stencils, choose the biasing according to the local advection velocity
                        if (u > 0) {
                            // Backward biased
                            H.F(iX, iY, iZ) -= u*gridData.xi_x(iX)*(-a*F.F(iX-1, iY, iZ) + b*F.F(iX, iY, iZ) - c*F.F(iX+1, iY, iZ))*i2dx;
                        } else {
                            // Forward biased
                            H.F(iX, iY, iZ) -= u*gridData.xi_x(iX)*(a*F.F(iX+1, iY, iZ) - b*F.F(iX, iY, iZ) + c*F.F(iX-1, iY, iZ))*i2dx;
                        }
                    }

                    u = V.Vy.F(iX, iY, iZ);
                    // First compute Peclet number
                    dh = gridData.y(iY+1) - gridData.y(iY);
                    pe = std::fabs(u)*dh/diffCoeff;

                    // If Peclet number is less than given limit, use central differencing, else biased stencils
                    if (pe < gridData.inputParams.peLimit) {
                        // Central difference
                        H.F(iX, iY, iZ) -= u*gridData.et_y(iY)*(F.F(iX, iY+1, iZ) - F.F(iX, iY-1, iZ))*i2dy;
                    } else {
                        // When using biased stencils, choose the biasing according to the local advection velocity
                        if (u > 0) {
                            // Backward biased
                            H.F(iX, iY, iZ) -= u*gridData.et_y(iY)*(-a*F.F(iX, iY-1, iZ) + b*F.F(iX, iY, iZ) - c*F.F(iX, iY+1, iZ))*i2dy;
                        } else {
                            // Forward biased
                            H.F(iX, iY, iZ) -= u*gridData.et_y(iY)*(a*F.F(iX, iY+1, iZ) - b*F.F(iX, iY, iZ) + c*F.F(iX, iY-1, iZ))*i2dy;
                        }
                    }

                    u = V.Vz.F(iX, iY, iZ);
                    // First compute Peclet number
                    dh = gridData.z(iZ+1) - gridData.z(iZ);
                    pe = std::fabs(u)*dh/diffCoeff;

                    // If Peclet number is less than given limit, use central differencing, else biased stencils
                    if (pe < gridData.inputParams.peLimit) {
                        // Central difference
                        H.F(iX, iY, iZ) -= u*gridData.zt_z(iZ)*(F.F(iX, iY, iZ+1) - F.F(iX, iY, iZ-1))*i2dz;
                    } else {
                        // When using biased stencils, choose the biasing according to the local advection velocity
                        if (u > 0) {
                            // Backward biased
                            H.F(iX, iY, iZ) -= u*gridData.zt_z(iZ)*(-a*F.F(iX, iY, iZ-1) + b*F.F(iX, iY, iZ) - c*F.F(iX, iY, iZ+1))*i2dz;
                        } else {
                            // Forward biased
                            H.F(iX, iY, iZ) -= u*gridData.zt_z(iZ)*(a*F.F(iX, iY, iZ+1) - b*F.F(iX, iY, iZ) + c*F.F(iX, iY, iZ-1))*i2dz;
                        }
                    }
                }
            }
        }
    } else {
        // SECOND ORDER UPWINDING
        // This uses second order forward and backward finite-difference stencils
        // mixed with second order central difference stencils where Pe is high.
        // Elsewhere it defaults to fourth order central-difference.
        for (int iX = 0; iX <= core.ubound(0); iX++) {
            for (int iY = 0; iY <= core.ubound(1); iY++) {
                for (int iZ = 0; iZ <= core.ubound(2); iZ++) {
                    u = V.Vx.F(iX, iY, iZ);
                    if (((iX == 0) and xfr) or ((iX == core.ubound(0)) and xlr)) {
                        // Central difference for first and last point
                        H.F(iX, iY, iZ) -= u*gridData.xi_x(iX)*(F.F(iX+1, iY, iZ) - F.F(iX-1, iY, iZ))*i2dx;
                    } else {
                        // First compute Peclet number
                        dh = gridData.x(iX+1) - gridData.x(iX);
                        pe = std::fabs(u)*dh/diffCoeff;

                        // If Peclet number is less than given limit, use central differencing, else biased stencils
                        if (pe < gridData.inputParams.peLimit) {
                            // Central difference
                            H.F(iX, iY, iZ) -= u*gridData.xi_x(iX)*(-F.F(iX+2, iY, iZ) + 8.0*F.F(iX+1, iY, iZ) - 8.0*F.F(iX-1, iY, iZ) + F.F(iX-2, iY, iZ))*i2dx/6.0;
                        } else {
                            // When using biased stencils, choose the biasing according to the local advection velocity
                            if (u > 0) {
                                // Backward biased
                                H.F(iX, iY, iZ) -= u*gridData.xi_x(iX)*(a*F.F(iX-2, iY, iZ) - b*F.F(iX-1, iY, iZ) + c*F.F(iX, iY, iZ) - d*F.F(iX+1, iY, iZ))*i2dx;
                            } else {
                                // Forward biased
                                H.F(iX, iY, iZ) -= u*gridData.xi_x(iX)*(-a*F.F(iX+2, iY, iZ) + b*F.F(iX+1, iY, iZ) - c*F.F(iX, iY, iZ) + d*F.F(iX-1, iY, iZ))*i2dx;
                            }
                        }
                    }

                    u = V.Vy.F(iX, iY, iZ);
                    if (((iY == 0) and yfr) or ((iY == core.ubound(1)) and ylr)) {
                        // Central difference for first and last point
                        H.F(iX, iY, iZ) -= u*gridData.et_y(iY)*(F.F(iX, iY+1, iZ) - F.F(iX, iY-1, iZ))*i2dy;
                    } else {
                        // First compute Peclet number
                        dh = gridData.y(iY+1) - gridData.y(iY);
                        pe = std::fabs(u)*dh/diffCoeff;

                        // If Peclet number is less than given limit, use central differencing, else biased stencils
                        if (pe < gridData.inputParams.peLimit) {
                            // Central difference
                            H.F(iX, iY, iZ) -= u*gridData.et_y(iY)*(-F.F(iX, iY+2, iZ) + 8.0*F.F(iX, iY+1, iZ) - 8.0*F.F(iX, iY-1, iZ) + F.F(iX, iY-2, iZ))*i2dy/6.0;
                        } else {
                            // When using biased stencils, choose the biasing according to the local advection velocity
                            if (u > 0) {
                                // Backward biased
                                H.F(iX, iY, iZ) -= u*gridData.et_y(iY)*(a*F.F(iX, iY-2, iZ) - b*F.F(iX, iY-1, iZ) + c*F.F(iX, iY, iZ) - d*F.F(iX, iY+1, iZ))*i2dy;
                            } else {
                                // Forward biased
                                H.F(iX, iY, iZ) -= u*gridData.et_y(iY)*(-a*F.F(iX, iY+2, iZ) + b*F.F(iX, iY+1, iZ) - c*F.F(iX, iY, iZ) + d*F.F(iX, iY-1, iZ))*i2dy;
                            }
                        }
                    }

                    u = V.Vz.F(iX, iY, iZ);
                    if (((iZ == 0) and zfr) or ((iZ == core.ubound(2)) and zlr)) {
                        // Central difference for first and last point
                        H.F(iX, iY, iZ) -= u*gridData.zt_z(iZ)*(F.F(iX, iY, iZ+1) - F.F(iX, iY, iZ-1))*i2dz;
                    } else {
                        // First compute Peclet number
                        dh = gridData.z(iZ+1) - gridData.z(iZ);
                        pe = std::fabs(u)*dh/diffCoeff;

                        // If Peclet number is less than given limit, use central differencing, else biased stencils
                        if (pe < gridData.inputParams.peLimit) {
                            // Central difference
                            H.F(iX, iY, iZ) -= u*gridData.zt_z(iZ)*(-F.F(iX, iY, iZ+2) + 8.0*F.F(iX, iY, iZ+1) - 8.0*F.F(iX, iY, iZ-1) + F.F(iX, iY, iZ-2))*i2dz/6.0;
                        } else {
                            // When using biased stencils, choose the biasing according to the local advection velocity
                            if (u > 0) {
                                // Backward biased
                                H.F(iX, iY, iZ) -= u*gridData.zt_z(iZ)*(a*F.F(iX, iY, iZ-2) - b*F.F(iX, iY, iZ-1) + c*F.F(iX, iY, iZ) - d*F.F(iX, iY, iZ+1))*i2dz;
                            } else {
                                // Forward biased
                                H.F(iX, iY, iZ) -= u*gridData.zt_z(iZ)*(-a*F.F(iX, iY, iZ+2) + b*F.F(iX, iY, iZ+1) - c*F.F(iX, iY, iZ) + d*F.F(iX, iY, iZ-1))*i2dz;
                            }
                        }
                    }
                }
            }
        }
    }
}

/**
 ********************************************************************************************************************************************
 * \brief   Operator to compute the gradient of the scalar field
 *
 *          The function computes the gradient of the cell centered scalar field, and stores it into a face-centered plainvf:
 *          \f$ \nabla f = \frac{\partial f}{\partial x}i + \frac{\partial f}{\partial y}j + \frac{\partial f}{\partial z}k \f$.
 *
 * \param   gradF is a reference to a plain vector field (plainvf) into which the computed gradient must be written.
 ********************************************************************************************************************************************
 */
void sfield::gradient(plainvf &gradF) {
    tempX = 0.0;
    derS.calcDerivative1_x(tempX);
    gradF.Vx(core) = tempX(core);
#ifndef PLANAR
    tempY = 0.0;
    derS.calcDerivative1_y(tempY);
    gradF.Vy(core) = tempY(core);
#endif
    tempZ = 0.0;
    derS.calcDerivative1_z(tempZ);
    gradF.Vz(core) = tempZ(core);
}

/**
 ********************************************************************************************************************************************
 * \brief   Function to impose the boundary conditions for the scalar field
 *
 *          The function first calls the \ref field#syncFaces "syncFaces" function of the field to update the sub-domain pads.
 *          Then the boundary conditions are applied at the full domain boundaries by calling the imposeBC()
 *          of each boundary class object assigned to each wall.
 *
 ********************************************************************************************************************************************
 */
void sfield::imposeBCs() {
    F.syncFaces();

    if (not gridData.inputParams.xPer) {
        tLft->imposeBC();
        tRgt->imposeBC();
    }
#ifndef PLANAR
    if (not gridData.inputParams.yPer) {
        tFrn->imposeBC();
        tBak->imposeBC();
    }
#endif
    if (not gridData.inputParams.zPer) {
        tTop->imposeBC();
        tBot->imposeBC();
    }
};

/**
 ********************************************************************************************************************************************
 * \brief   Function to set the diffusion coefficient of the scalar field
 *
 *          The function sets the diffusion coefficient corresponding to the scalar field.
 *          This constant is used when computing the Peclet number when upwinding is enabled.
 *          Trying to use upwinding without setting this constant may cause the code to crash.
 *
 ********************************************************************************************************************************************
 */
void sfield::setDiffCoeff(real dCoeff) {
    diffCoeff = dCoeff;
}

/**
 ********************************************************************************************************************************************
 * \brief   Overloaded operator to add a given plain scalar field
 *
 *          The unary operator += adds a given plain scalar field to the sfield and returns
 *          a pointer to itself.
 *
 * \param   a is a reference to a plainsf to be added to the member field
 *
 * \return  A pointer to itself is returned by the scalar field class to which the operator belongs
 ********************************************************************************************************************************************
 */
sfield& sfield::operator += (plainsf &a) {
    F.F += a.F;

    return *this;
}

/**
 ********************************************************************************************************************************************
 * \brief   Overloaded operator to subtract a given plain scalar field
 *
 *          The unary operator -= subtracts a given plain scalar field from the sfield and returns
 *          a pointer to itself.
 *
 * \param   a is a reference to a plainsf to be subtracted from the member field
 *
 * \return  A pointer to itself is returned by the scalar field class to which the operator belongs
 ********************************************************************************************************************************************
 */
sfield& sfield::operator -= (plainsf &a) {
    F.F -= a.F;

    return *this;
}

/**
 ********************************************************************************************************************************************
 * \brief   Overloaded operator to add a given scalar field
 *
 *          The unary operator += adds a given scalar field to the sfield and returns
 *          a pointer to itself.
 *
 * \param   a is a reference to another sfield to be added to the member field
 *
 * \return  A pointer to itself is returned by the scalar field class to which the operator belongs
 ********************************************************************************************************************************************
 */
sfield& sfield::operator += (sfield &a) {
    F.F += a.F.F;

    return *this;
}

/**
 ********************************************************************************************************************************************
 * \brief   Overloaded operator to subtract a given scalar field
 *
 *          The unary operator -= subtracts a given scalar field from the sfield and returns
 *          a pointer to itself.
 *
 * \param   a is a reference to another sfield to be subtracted from the member field
 *
 * \return  A pointer to itself is returned by the scalar field class to which the operator belongs
 ********************************************************************************************************************************************
 */
sfield& sfield::operator -= (sfield &a) {
    F.F -= a.F.F;

    return *this;
}

/**
 ********************************************************************************************************************************************
 * \brief   Overloaded operator to multiply a scalar value to the scalar field
 *
 *          The unary operator *= multiplies a real value to the sfield and returns
 *          a pointer to itself.
 *
 * \param   a is a real number to be multiplied to the scalar field
 *
 * \return  A pointer to itself is returned by the scalar field class to which the operator belongs
 ********************************************************************************************************************************************
 */
sfield& sfield::operator *= (real a) {
    F.F *= a;

    return *this;
}

/**
 ********************************************************************************************************************************************
 * \brief   Overloaded operator to assign a plain scalar field to the scalar field
 *
 *          The operator = copies the contents of the input plain scalar field to itself.
 *
 * \param   a is the plainsf to be assigned to the scalar field
 ********************************************************************************************************************************************
 */
void sfield::operator = (plainsf &a) {
    F.F = a.F;
}

/**
 ********************************************************************************************************************************************
 * \brief   Overloaded operator to assign a scalar field to the scalar field
 *
 *          The operator = copies the contents of the input scalar field to itself.
 *
 * \param   a is the scalar field to be assigned to the scalar field
 ********************************************************************************************************************************************
 */
void sfield::operator = (sfield &a) {
    F.F = a.F.F;
}

/**
 ********************************************************************************************************************************************
 * \brief   Overloaded operator to assign a scalar value to the scalar field
 *
 *          The operator = assigns a real value to all the scalar field.
 *
 * \param   a is a real number to be assigned to the scalar field
 ********************************************************************************************************************************************
 */
void sfield::operator = (real a) {
    F.F = a;
}
