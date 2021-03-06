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
/*! \file periodic.cc
 *
 *  \brief Definitions for functions of class boundary
 *  \sa boundary.h
 *  \author Roshan Samuel
 *  \date Nov 2019
 *  \copyright New BSD License
 *
 ********************************************************************************************************************************************
 */

#include "boundary.h"

/**
 ********************************************************************************************************************************************
 * \brief   Constructor of the periodic class
 *
 *          The constructor simply initializes the base boundary class using all the arguments supplied to it.
 *          Since periodic BC is being implemented, no additional values are necessary for the object.
 *
 * \param   mesh is a const reference to the global data contained in the grid class
 * \param   inField is a reference to the field to which the boundary conditions must be applied.
 * \param   bcWall is a const integer which specifies the wall to which the BC must be applied.
 ********************************************************************************************************************************************
 */
periodic::periodic(const grid &mesh, field &inField, const int bcWall):
                            boundary(mesh, inField, bcWall) {
    // If shiftVal = 1, the wall is either left (0), front (2), or bottom (4) wall
    // In this case, the data next to the opposite wall (wallNum + 1) as to be used
    // If shiftVal = -1, the wall is either right (1), back (3), or top (5) wall
    // In this case, the opposite wall is wallNum - 1
    dataSlice = mesh.shift(shiftDim, dField.fWalls(wallNum + shiftVal), -shiftVal);
}

/**
 ********************************************************************************************************************************************
 * \brief   Function to impose periodic BC on a cell centered variable
 *
 *          For Saras solver, all variables are at cell-centers, while the walls pass along
 *          the faces of the cells.
 *          Hence the ghost point and adjacent point just inside the domain are lying on either side of the wall.
 *          This BC is used only when running a periodic simulation on a single core.
 *          When using multiple cores, the MPI data-transfer normally handles this BC.
 *
 ********************************************************************************************************************************************
 */
inline void periodic::imposeBC() {
    // The BC is applied for all ranks and no rankFlag is used
    dField.F(wallSlice) = dField.F(dataSlice);
}
