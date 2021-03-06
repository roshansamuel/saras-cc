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
/*! \file derivative.h
 *
 *  \brief Class declaration of derivative
 *
 *  \author Roshan Samuel, Ali Asad
 *  \date Nov 2019
 *  \copyright New BSD License
 *
 ********************************************************************************************************************************************
 */

#ifndef DERIVATIVE_H
#define DERIVATIVE_H

#include <blitz/array.h>
#include <blitz/array/stencil-et.h>
#include <blitz/array/stencilops.h>

#include "grid.h"

class derivative {
    private: 
        const grid &gridData;

        const blitz::Array<real, 3> &F;

        /** Flags for first rank (fr) and last rank (lr) along X, Y and Z directions */
        bool xfr, xlr, yfr, ylr, zfr, zlr;

        real ihx, ihy, ihz;
        real ihx2, ihy2, ihz2;

        blitz::firstIndex i;
        blitz::secondIndex j;
        blitz::thirdIndex k;    

        blitz::RectDomain<3> core;

        blitz::RectDomain<3> x0Lft, x0Mid, x0Rgt;
        blitz::RectDomain<3> x1Lft, x1Mid, x1Rgt;

        blitz::RectDomain<3> y0Lft, y0Mid, y0Rgt;
        blitz::RectDomain<3> y1Lft, y1Mid, y1Rgt;

        blitz::RectDomain<3> z0Lft, z0Mid, z0Rgt;
        blitz::RectDomain<3> z1Lft, z1Mid, z1Rgt;

        blitz::Range fullRange;
        blitz::Range xRange, yRange, zRange;

        blitz::Array<real, 3> tmpArray;

        void setWallRectDomains();

    public:
        derivative(const grid &gridData, const blitz::Array<real, 3> &F);

        void calcDerivative1_x(blitz::Array<real, 3> outArray);
        void calcDerivative1_y(blitz::Array<real, 3> outArray);
        void calcDerivative1_z(blitz::Array<real, 3> outArray);

        void calcDerivative2xx(blitz::Array<real, 3> outArray);
        void calcDerivative2yy(blitz::Array<real, 3> outArray);
        void calcDerivative2zz(blitz::Array<real, 3> outArray);

        ~derivative() { };
};

/**
 ********************************************************************************************************************************************
 *  \class derivative derivative.h "lib/derivative.h"
 *  \brief Derivative class to perform finite difference operations on the data stored in field
 *
 *  It contains functions to perform the finite difference operations on fields.
 *  For fields on non-uniform grids, the derivatives are transformed with appropriate
 *  grid derivatives taken from the grid class.
 *  For many classes of SARAS, empty destructors were removed.
 *  Refer reference [3] of General Articles in README for more details.
 ********************************************************************************************************************************************
 */

#endif
