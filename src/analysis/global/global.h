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
/*! \file global.h
 *
 *  \brief Declarations of global functions and variables for post-processing
 *
 *  \author Roshan Samuel
 *  \date May 2022
 *  \copyright New BSD License
 *
 ********************************************************************************************************************************************
 */

#ifndef POSTGLOBAL_H
#define POSTGLOBAL_H

#include <blitz/array.h>
#include <iostream>

#include "vfield.h"
#include "sfield.h"

class global {
    private:
        // Loop limits for mid-point method of integration
        int xlMP, xuMP, ylMP, yuMP, zlMP, zuMP;

        /** Flags for first rank (fr) and last rank (lr) along X, Y and Z directions */
        bool xfr, xlr, yfr, ylr, zfr, zlr;

        blitz::RectDomain<3> x0Lft, x0Rgt, x1Lft, x1Rgt;
        blitz::RectDomain<3> y0Lft, y0Rgt, y1Lft, y1Rgt;
        blitz::RectDomain<3> z0Lft, z0Rgt, z1Lft, z1Rgt;

        real simpsonBase(blitz::Array<real, 3> F, blitz::Array<real, 1> Z, blitz::Array<real, 1> Y, blitz::Array<real, 1> X, int sIndex, int eIndex);
        real simpsonRule(blitz::Array<real, 3> F, blitz::Array<real, 1> Z, blitz::Array<real, 1> Y, blitz::Array<real, 1> X);

        real simpsonBase(blitz::Array<real, 2> F, blitz::Array<real, 1> Y, blitz::Array<real, 1> X, int sIndex, int eIndex);
        real simpsonRule(blitz::Array<real, 2> F, blitz::Array<real, 1> Y, blitz::Array<real, 1> X);

        real simpsonBase(blitz::Array<real, 1> F, blitz::Array<real, 1> X, int sIndex, int eIndex);
        real simpsonRule(blitz::Array<real, 1> F, blitz::Array<real, 1> X);
    public:
        /** A const reference to the global variables stored in the grid class */
        const grid &mesh;

        global(const grid &mesh);

        void setWallRectDomains();
        blitz::Array<real, 3> shift2Wall(blitz::Array<real, 3> F);

        // These are copies of the same functions in the hydro/scalar classes.
        // A better method of using the same function without repeating code
        // needs to be implemented in future.
        void initVBCs(vfield &V);
        void initTBCs(sfield &T);
        void checkPeriodic(const parser &inputParams, parallel &rankData);

        real simpsonInt(blitz::Array<real, 3> F, blitz::Array<real, 1> Z, blitz::Array<real, 1> Y, blitz::Array<real, 1> X);
        real simpsonInt(blitz::Array<real, 2> F, blitz::Array<real, 1> Y, blitz::Array<real, 1> X);
        real simpsonInt(blitz::Array<real, 1> F, blitz::Array<real, 1> X);

        real volAvgMidPt(blitz::Array<real, 3> F);
};

/**
 ********************************************************************************************************************************************
 *  \class global global.h "src/analysis/global/global.h"
 *  \brief  Contains all the global variables related to post-processing.
 ********************************************************************************************************************************************
 */

#endif
