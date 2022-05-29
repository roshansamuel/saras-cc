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
#include "sfield.h"

static void taylorGreen(sfield &F, sfield &dF, grid &mesh);

void dissipation(grid &gridData, std::vector<real> tList) {
    sfield F(gridData, "F");
    sfield dF(gridData, "dF");

    if (gridData.pf) std::cout << tList.size() << std::endl;
    taylorGreen(F, dF, gridData);
}

static void taylorGreen(sfield &F, sfield &dF, grid &mesh) {
    for (int i=F.F.F.lbound(0); i <= F.F.F.ubound(0); i++) {
        for (int j=F.F.F.lbound(1); j <= F.F.F.ubound(1); j++) {
            for (int k=F.F.F.lbound(2); k <= F.F.F.ubound(2); k++) {
                F.F.F(i, j, k) = sin(2.0*M_PI*mesh.x(i)/mesh.xLen)*
                                 cos(2.0*M_PI*mesh.y(j)/mesh.yLen)*
                                 cos(2.0*M_PI*mesh.z(k)/mesh.zLen);
                dF.F.F(i, j, k) = 2.0*M_PI*cos(2.0*M_PI*mesh.x(i)/mesh.xLen)*
                                           cos(2.0*M_PI*mesh.y(j)/mesh.yLen)*
                                           cos(2.0*M_PI*mesh.z(k)/mesh.zLen);
            }
        }
    }
}
