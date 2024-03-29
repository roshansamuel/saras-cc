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
/*! \file parallel.h
 *
 *  \brief Class declaration of parallel
 *
 *  \author Roshan Samuel
 *  \date Nov 2019
 *  \copyright New BSD License
 *
 ********************************************************************************************************************************************
 */

#ifndef PARALLEL_H
#define PARALLEL_H

#include <mpi.h>

#include "parser.h"

class parallel {
    private:
        inline void assignRanks();
        void getNeighbours();

    public:
        // ALL THE INTEGERS USED BELOW ARE POSITIVE. STILL IT IS BETTER TO USE int INSTEAD OF unsigned int
        /** The MPI rank of each sub-domain */
        int rank;

        /** The total number of cores available for computation */
        int nProc;

        /** npX, npY and npZ indicates the number of sub-domain divisions along the X, Y and Z directions respectively */
        //@{
        const int npX, npY, npZ;
        //@}

        /** xRank, yRank and zRank indicates the rank in terms of sub-domain divisions along the X, Y and Z directions respectively.
         *  Like the global rank variable, these values also start from 0 to npX - 1, npY - 1, and npZ - 1 respectively. */
        //@{
        int xRank, yRank, zRank;
        //@}

        /** Array of ranks of the 6 neighbouring sub-domains across faces:
          * Left, Right, Front, Back, Top, Bottom
          * */
        blitz::Array<int, 1> faceRanks;

        /** Array of ranks of the 12 neighbouring sub-domains across edges:
          * Left-Front, Left-Back, Right-Front, Right-Back
          * Front-Bottom, Front-Top, Back-Bottom, Back-Top
          * Bottom-Left, Bottom-Right, Top-Left, Top-Right
          * */
        blitz::Array<int, 1> edgeRanks;

        /** Array of ranks of the 8 neighbouring sub-domains across corners:
          * Left-Front-Bottom, Left-Back-Bottom, Right-Front-Bottom, Right-Back-Bottom
          * Left-Front-Top, Left-Back-Top, Right-Front-Top, Right-Back-Top
          * */
        blitz::Array<int, 1> cornRanks;

        parallel(const parser &iDat);

/**
 ********************************************************************************************************************************************
 * \brief   Function to calculate the positive modulus of two numbers
 *
 *          The inline function return the positive modulus of 2 numbers, with negative values
 *          wrapping around to the upper limit.
 *
 *
 * \param   a is the integer first operand as in ordinary mod function
 * \param   b is the integer second operand as in ordinary mod function
 *
 * \return  The integer value of the positive modulus of the two input numbers
 ********************************************************************************************************************************************
 */

        static inline int pmod(int a, int b) {return (a % b + b) % b;};

/**
 ********************************************************************************************************************************************
 * \brief   Function to calculate the global rank of a sub-domain using its xRank, yRank and zRank
 *
 *          The inline function computes the global rank of the processor using xRank, yRank and zRank.
 *          In doing so, a periodic domain is assumed. Non-periodic problems must have ranks set specifically.
 *
 *
 * \param   xR is the integer value of the sub-domain's xRank
 * \param   yR is the integer value of the sub-domain's yRank
 * \param   zR is the integer value of the sub-domain's zRank
 *
 * \return  The integer value of the rank of the sub-domain
 ********************************************************************************************************************************************
 */

        inline int findRank(int xR, int yR, int zR) const {return pmod(zR, npZ)*npX*npY + pmod(yR, npY)*npX + pmod(xR, npX);};
};

/**
 ********************************************************************************************************************************************
 *  \class parallel parallel.h "lib/parallel.h"
 *  \brief Class for all the global variables and functions related to parallelization.
 *
 *  After MPI_Init, every process has its own rank. Moreover, after performing domain decomposition, each process has its own
 *  xRank and yRank to identify its position within the global computational domain.
 *  These data, along with the data to identify the neighbouring processes for inter-domain communication are stored in the
 *  <B>parallel</B> class.
 *  This class is initialized only once at the start of the solver.
 ********************************************************************************************************************************************
 */

#endif
