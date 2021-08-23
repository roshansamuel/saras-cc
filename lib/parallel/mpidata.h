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
/*! \file mpidata.h
 *
 *  \brief Class declaration of mpidata
 *
 *  \author Roshan Samuel
 *  \date Nov 2019
 *  \copyright New BSD License
 *
 ********************************************************************************************************************************************
 */

#ifndef MPIDATA_H
#define MPIDATA_H

#include "parallel.h"

class mpidata {
    private:
        /** MPI derived datatypes for data to be sent/received across the faces of sub-domains */
        blitz::Array<MPI_Datatype, 1> fsSubs, frSubs;

        /** MPI derived datatypes for data to be sent/received across the edges of sub-domains */
        blitz::Array<MPI_Datatype, 1> esSubs, erSubs;

        /** MPI derived datatypes for data to be sent/received across the corners of sub-domains */
        blitz::Array<MPI_Datatype, 1> csSubs, crSubs;

        /** Array of tags for receiving data. */
        blitz::Array<int, 1> fTags, eTags, cTags;

        /** An array of MPI_Request data-types necessary for obtaining output from the non-blocking receive MPI_Irecv in the synchronizing functions. */
        blitz::Array<MPI_Request, 1> recvRequest;

        /** An array of MPI_Status data-types necessary for obtaining output from the non-blocking receive MPI_Irecv in the synchronizing functions. */
        blitz::Array<MPI_Status, 1> recvStatus;

        /** Blitz array of the data field which needs to be synchronised across processors. */
        blitz::Array<real, 3> dataField;

    public:
        /** A const reference to the global variables stored in the parallel class to access rank data */
        const parallel &rankData;

        mpidata(blitz::Array<real, 3> inputArray, const parallel &parallelData);

        void createSubarrays(const blitz::TinyVector<int, 3> globSize,
                             const blitz::TinyVector<int, 3> coreSize,
                             const blitz::TinyVector<int, 3> padWidth);

        void syncFaces();
        void syncAll();
};

/**
 ********************************************************************************************************************************************
 *  \class mpidata mpidata.h "lib/mpidata.h"
 *  \brief Class to store MPI derived datatypes for individual arrays.
 *
 *  Since the solver uses staggered and collocated grids, the data of its variables are stored in arrays of different limits
 *  depending on whether the variable is staggered or not in each direction.
 *  As a result, the limits of the sub-arrays to be sent across inter-processor boundaries is different for different arrays.
 *  Hence the <B>mpidata</B> class contains MPI_SUBARRAY derived datatypes to be initialized along with different fields in order
 *  to store their sub-arrays for inter-processor communication.
 ********************************************************************************************************************************************
 */

#endif
