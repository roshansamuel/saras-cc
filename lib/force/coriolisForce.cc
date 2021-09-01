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
/*! \file coriolisForce.cc
 *
 *  \brief Definitions for functions of class coriolisForce
 *  \sa force.h
 *  \author Shashwat Bhattacharya, Roshan Samuel
 *  \date Nov 2019
 *  \copyright New BSD License
 *
 ********************************************************************************************************************************************
 */

#include "force.h"

coriolisForce::coriolisForce(const grid &mesh, const vfield &U): force(mesh, U) {
    real vNorm;

    Fr = 1.0/mesh.inputParams.Ro;

    vNorm = std::sqrt(blitz::sum(blitz::sqr(mesh.inputParams.rAxis)));
    rTerms = Fr*mesh.inputParams.rAxis/vNorm;
}


void coriolisForce::addForcing(plainvf &Hv) {
    // ADD THE ROTATING TERM TO THE CORRESPONDING COMPONENTS OF Hv
    // THE TERMS ARE CALCULATED AS n x v,
    // WHERE n IS THE UNIT VECTOR ALONG ROTATION AXIS, AND v IS VELOCITY VECTOR.
    Hv.Vx -= rTerms[1]*V.Vz.F - rTerms[2]*V.Vy.F;
    Hv.Vy -= rTerms[2]*V.Vx.F - rTerms[0]*V.Vz.F;
    Hv.Vz -= rTerms[0]*V.Vy.F - rTerms[1]*V.Vx.F;
}
