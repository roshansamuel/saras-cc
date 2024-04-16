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
/*! \file plumeOut.cc
 *
 *  \brief Definitions for functions of class boundary
 *  \sa boundary.h
 *  \author Roshan Samuel
 *  \date Mar 2024
 *  \copyright New BSD License
 *
 ********************************************************************************************************************************************
 */

#include "boundary.h"

/**
 ********************************************************************************************************************************************
 * \brief   Constructor of the plumeOut class
 *
 *          The constructor initializes the base boundary class using part of the arguments supplied to it.
 *          The plume outflow BC is used for 2D simulation of plume rising according to reference [1]
 *
 * \param   mesh is a const reference to the global data contained in the grid class
 * \param   inField is a reference to scalar field to which the boundary conditions must be applied.
 * \param   bcWall is a const integer which specifies the wall to which the BC must be applied.
 * \param   wField is the vertical velocity field used to toggle between dirichlet and neumann BC.
 ********************************************************************************************************************************************
 */
plumeOut::plumeOut(const grid &mesh, field &inField, const int bcWall, field &wField):
                            boundary(mesh, inField, bcWall), maskField(wField) { }


/**
 ********************************************************************************************************************************************
 * \brief   Function to impose Mixed BC of a heating plate on a cell centered variable
 *
 *          This form of boundary condition for allowing plumes to go out of the top wall
 *          is derived from the reference [1].
 *
 ********************************************************************************************************************************************
 */
void plumeOut::imposeBC() {
    for (int iX = dField.fWalls(wallNum).lbound(0); iX <= dField.fWalls(wallNum).ubound(0); iX++) {
        for (int iY = dField.fWalls(wallNum).lbound(1); iY <= dField.fWalls(wallNum).ubound(1); iY++) {
            for (int iZ = dField.fWalls(wallNum).lbound(2); iZ <= dField.fWalls(wallNum).ubound(2); iZ++) {
                if (maskField.F(iX, iY, iZ) > 0) {
                    dField.F(iX, iY, iZ) = dField.F(iX, iY, iZ-1);
                } else {
                    dField.F(iX, iY, iZ) = 0.0;
                }
            }
        }
    }
}

/*
 * REFERENCES
 * [1]. Theerthan A. and Arakeri J. H., Model for near-wall dynamics in turbulent RBC, J. Fluid Mech., 373 (1998)
 */
