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
/*! \file mpidata.cc
 *
 *  \brief Definitions for functions of class mpidata
 *  \sa mpidata.h
 *  \author Roshan Samuel
 *  \date Nov 2019
 *  \copyright New BSD License
 *
 ********************************************************************************************************************************************
 */

#include "mpidata.h"

/**
 ********************************************************************************************************************************************
 * \brief   Constructor of the mpidata class
 *
 *          The short constructor of mpidata class merely resizes the array of MPI_Status and MPI_Request datatypes.
 *          The former is used in non-blocking communication of MPI_Irecv, while the later is used in the MPI_Waitall
 *          function to complete the non-blocking communication call.
 *
 * \param   inputArray is the blitz array whose sub-arrays have to be created and synchronised across processors
 * \param   parallelData is a const reference to the global data contained in the parallel class
 ********************************************************************************************************************************************
 */
mpidata::mpidata(blitz::Array<real, 3> inputArray, const parallel &parallelData): dataField(inputArray), rankData(parallelData) {
    recvStatus.resize(12);
    recvRequest.resize(12);

    fsSubs.resize(6);
    frSubs.resize(6);
    fTags.resize(6);

    fTags = 2, 1, 4, 3, 6, 5;

    esSubs.resize(12);
    erSubs.resize(12);
    eTags.resize(12);

    eTags = 4, 3, 2, 1, 8, 7, 6, 5, 12, 11, 10, 9;

    csSubs.resize(8);
    crSubs.resize(8);
    cTags.resize(8);

    cTags = 8, 7, 6, 5, 4, 3, 2, 1;
}

/**
 ********************************************************************************************************************************************
 * \brief   Function to create the subarray MPI_Datatypes
 *
 *          Must be called only after the grid class has been initialized.
 *          The subarray data-types cannot be created within the constructor of the parallel class as it needs the grid parameters for
 *          setting the limits of the subarrays.
 *          For this, the grid class will have to be included in the parallel class.
 *
 *          However, the grid object cannot be passed to the parallel class as the grid class already includes the parallel object
 *          within itself, and a reverse include will raise cyclic dependency error.
 *          As a result, the mpidata class offers an additional layer over the parallel class for grid specific data transfer functions.
 *
 * \param   globSize stores the global size of a sub-domain - including core and pads
 * \param   coreSize stores the size of the core of the sub-domain and is similar to the collocCoreSize variable in the grid class
 * \param   padWidth contains the widths of pads along the 3 directions, namely padWidths TinyVector from the grid class
 ********************************************************************************************************************************************
 */
void mpidata::createSubarrays(const blitz::TinyVector<int, 3> globSize,
                              const blitz::TinyVector<int, 3> coreSize,
                              const blitz::TinyVector<int, 3> padWidth) {
    blitz::TinyVector<int, 3> globCopy;
    blitz::TinyVector<int, 3> loclSize;
    blitz::TinyVector<int, 3> saStarts;

    globCopy = globSize;

    /********************************************************************************************************/
    /************************ MPI SUB-ARRAY DATATYPES FOR DATA TRANSFER ACROSS FACES ************************/
    /********************************************************************************************************/

    /** ALONG X-DIRECTION **/

    loclSize = coreSize;            loclSize(0) = padWidth(0);

    /////// LEFT FACE ///////
    saStarts = padWidth;
    MPI_Type_create_subarray(3, globCopy.data(), loclSize.data(), saStarts.data(), MPI_ORDER_C, MPI_FP_REAL, &fsSubs(0));

    saStarts = padWidth;            saStarts(0) = 0;
    MPI_Type_create_subarray(3, globCopy.data(), loclSize.data(), saStarts.data(), MPI_ORDER_C, MPI_FP_REAL, &frSubs(0));

    /////// RIGHT FACE //////
    saStarts = padWidth;            saStarts(0) = coreSize(0);
    MPI_Type_create_subarray(3, globCopy.data(), loclSize.data(), saStarts.data(), MPI_ORDER_C, MPI_FP_REAL, &fsSubs(1));

    saStarts = padWidth;            saStarts(0) = coreSize(0) + padWidth(0);
    MPI_Type_create_subarray(3, globCopy.data(), loclSize.data(), saStarts.data(), MPI_ORDER_C, MPI_FP_REAL, &frSubs(1));


    /** ALONG Y-DIRECTION **/

    loclSize = coreSize;            loclSize(1) = padWidth(1);

    /////// FRONT FACE //////
    saStarts = padWidth;
    MPI_Type_create_subarray(3, globCopy.data(), loclSize.data(), saStarts.data(), MPI_ORDER_C, MPI_FP_REAL, &fsSubs(2));

    saStarts = padWidth;            saStarts(1) = 0;
    MPI_Type_create_subarray(3, globCopy.data(), loclSize.data(), saStarts.data(), MPI_ORDER_C, MPI_FP_REAL, &frSubs(2));

    /////// BACK FACE ///////
    saStarts = padWidth;            saStarts(1) = coreSize(1);
    MPI_Type_create_subarray(3, globCopy.data(), loclSize.data(), saStarts.data(), MPI_ORDER_C, MPI_FP_REAL, &fsSubs(3));

    saStarts = padWidth;            saStarts(1) = coreSize(1) + padWidth(1);
    MPI_Type_create_subarray(3, globCopy.data(), loclSize.data(), saStarts.data(), MPI_ORDER_C, MPI_FP_REAL, &frSubs(3));


    /** ALONG Z-DIRECTION **/

    loclSize = coreSize;            loclSize(2) = padWidth(2);

    ////// BOTTOM FACE //////
    saStarts = padWidth;
    MPI_Type_create_subarray(3, globCopy.data(), loclSize.data(), saStarts.data(), MPI_ORDER_C, MPI_FP_REAL, &fsSubs(4));

    saStarts = padWidth;            saStarts(2) = 0;
    MPI_Type_create_subarray(3, globCopy.data(), loclSize.data(), saStarts.data(), MPI_ORDER_C, MPI_FP_REAL, &frSubs(4));

    /////// TOP FACE ////////
    saStarts = padWidth;            saStarts(2) = coreSize(2);
    MPI_Type_create_subarray(3, globCopy.data(), loclSize.data(), saStarts.data(), MPI_ORDER_C, MPI_FP_REAL, &fsSubs(5));

    saStarts = padWidth;            saStarts(2) = coreSize(2) + padWidth(2);
    MPI_Type_create_subarray(3, globCopy.data(), loclSize.data(), saStarts.data(), MPI_ORDER_C, MPI_FP_REAL, &frSubs(5));


    /********************************************************************************************************/
    /************************ MPI SUB-ARRAY DATATYPES FOR DATA TRANSFER ACROSS EDGES ************************/
    /********************************************************************************************************/

    /** ALONG Z-DIRECTION **/

    loclSize = padWidth;            loclSize(2) = coreSize(2);

    /////// LEFT-FRONT EDGE ///////
    saStarts = padWidth;
    MPI_Type_create_subarray(3, globCopy.data(), loclSize.data(), saStarts.data(), MPI_ORDER_C, MPI_FP_REAL, &esSubs(0));

    saStarts = 0, 0, 0;             saStarts(2) = padWidth(2);
    MPI_Type_create_subarray(3, globCopy.data(), loclSize.data(), saStarts.data(), MPI_ORDER_C, MPI_FP_REAL, &erSubs(0));

    /////// LEFT-BACK EDGE ////////
    saStarts = padWidth;            saStarts(1) = coreSize(1);
    MPI_Type_create_subarray(3, globCopy.data(), loclSize.data(), saStarts.data(), MPI_ORDER_C, MPI_FP_REAL, &esSubs(1));

    saStarts = 0, 0, 0;             saStarts(1) = coreSize(1) + padWidth(1);            saStarts(2) = padWidth(2);
    MPI_Type_create_subarray(3, globCopy.data(), loclSize.data(), saStarts.data(), MPI_ORDER_C, MPI_FP_REAL, &erSubs(1));

    ////// RIGHT-FRONT EDGE ///////
    saStarts = padWidth;            saStarts(0) = coreSize(0);
    MPI_Type_create_subarray(3, globCopy.data(), loclSize.data(), saStarts.data(), MPI_ORDER_C, MPI_FP_REAL, &esSubs(2));

    saStarts = 0, 0, 0;             saStarts(0) = coreSize(0) + padWidth(0);            saStarts(2) = padWidth(2);
    MPI_Type_create_subarray(3, globCopy.data(), loclSize.data(), saStarts.data(), MPI_ORDER_C, MPI_FP_REAL, &erSubs(2));

    /////// RIGHT-BACK EDGE ///////
    saStarts = coreSize;            saStarts(2) = padWidth(2);
    MPI_Type_create_subarray(3, globCopy.data(), loclSize.data(), saStarts.data(), MPI_ORDER_C, MPI_FP_REAL, &esSubs(3));

    saStarts = coreSize + padWidth; saStarts(2) = padWidth(2);
    MPI_Type_create_subarray(3, globCopy.data(), loclSize.data(), saStarts.data(), MPI_ORDER_C, MPI_FP_REAL, &erSubs(3));


    /** ALONG X-DIRECTION **/

    loclSize = padWidth;            loclSize(0) = coreSize(0);

    ////// FRONT-BOTTOM EDGE //////
    saStarts = padWidth;
    MPI_Type_create_subarray(3, globCopy.data(), loclSize.data(), saStarts.data(), MPI_ORDER_C, MPI_FP_REAL, &esSubs(4));

    saStarts = 0, 0, 0;             saStarts(0) = padWidth(0);
    MPI_Type_create_subarray(3, globCopy.data(), loclSize.data(), saStarts.data(), MPI_ORDER_C, MPI_FP_REAL, &erSubs(4));

    /////// FRONT-TOP EDGE ////////
    saStarts = padWidth;            saStarts(2) = coreSize(2);
    MPI_Type_create_subarray(3, globCopy.data(), loclSize.data(), saStarts.data(), MPI_ORDER_C, MPI_FP_REAL, &esSubs(5));

    saStarts = 0, 0, 0;             saStarts(2) = coreSize(2) + padWidth(2);            saStarts(0) = padWidth(0);
    MPI_Type_create_subarray(3, globCopy.data(), loclSize.data(), saStarts.data(), MPI_ORDER_C, MPI_FP_REAL, &erSubs(5));

    /////// BACK-BOTTOM EDGE //////
    saStarts = padWidth;            saStarts(1) = coreSize(1);
    MPI_Type_create_subarray(3, globCopy.data(), loclSize.data(), saStarts.data(), MPI_ORDER_C, MPI_FP_REAL, &esSubs(6));

    saStarts = 0, 0, 0;             saStarts(1) = coreSize(1) + padWidth(1);            saStarts(0) = padWidth(0);
    MPI_Type_create_subarray(3, globCopy.data(), loclSize.data(), saStarts.data(), MPI_ORDER_C, MPI_FP_REAL, &erSubs(6));

    ///////// BACK-TOP EDGE ///////
    saStarts = coreSize;            saStarts(0) = padWidth(0);
    MPI_Type_create_subarray(3, globCopy.data(), loclSize.data(), saStarts.data(), MPI_ORDER_C, MPI_FP_REAL, &esSubs(7));

    saStarts = coreSize + padWidth; saStarts(0) = padWidth(0);
    MPI_Type_create_subarray(3, globCopy.data(), loclSize.data(), saStarts.data(), MPI_ORDER_C, MPI_FP_REAL, &erSubs(7));


    /** ALONG Y-DIRECTION **/

    loclSize = padWidth;            loclSize(1) = coreSize(1);

    ////// BOTTOM-LEFT EDGE ///////
    saStarts = padWidth;
    MPI_Type_create_subarray(3, globCopy.data(), loclSize.data(), saStarts.data(), MPI_ORDER_C, MPI_FP_REAL, &esSubs(8));

    saStarts = 0, 0, 0;             saStarts(1) = padWidth(1);
    MPI_Type_create_subarray(3, globCopy.data(), loclSize.data(), saStarts.data(), MPI_ORDER_C, MPI_FP_REAL, &erSubs(8));

    ////// BOTTOM-RIGHT EDGE //////
    saStarts = padWidth;            saStarts(0) = coreSize(0);
    MPI_Type_create_subarray(3, globCopy.data(), loclSize.data(), saStarts.data(), MPI_ORDER_C, MPI_FP_REAL, &esSubs(9));

    saStarts = 0, 0, 0;             saStarts(0) = coreSize(0) + padWidth(0);            saStarts(1) = padWidth(1);
    MPI_Type_create_subarray(3, globCopy.data(), loclSize.data(), saStarts.data(), MPI_ORDER_C, MPI_FP_REAL, &erSubs(9));

    //////// TOP-LEFT EDGE ////////
    saStarts = padWidth;            saStarts(2) = coreSize(2);
    MPI_Type_create_subarray(3, globCopy.data(), loclSize.data(), saStarts.data(), MPI_ORDER_C, MPI_FP_REAL, &esSubs(10));

    saStarts = 0, 0, 0;             saStarts(2) = coreSize(2) + padWidth(2);            saStarts(1) = padWidth(1);
    MPI_Type_create_subarray(3, globCopy.data(), loclSize.data(), saStarts.data(), MPI_ORDER_C, MPI_FP_REAL, &erSubs(10));

    //////// TOP-RIGHT EDGE ///////
    saStarts = coreSize;            saStarts(1) = padWidth(1);
    MPI_Type_create_subarray(3, globCopy.data(), loclSize.data(), saStarts.data(), MPI_ORDER_C, MPI_FP_REAL, &esSubs(11));

    saStarts = coreSize + padWidth; saStarts(1) = padWidth(1);
    MPI_Type_create_subarray(3, globCopy.data(), loclSize.data(), saStarts.data(), MPI_ORDER_C, MPI_FP_REAL, &erSubs(11));


    /********************************************************************************************************/
    /*********************** MPI SUB-ARRAY DATATYPES FOR DATA TRANSFER ACROSS CORNERS ***********************/
    /********************************************************************************************************/

    loclSize = padWidth;

    /// LEFT-FRONT-BOTTOM CORNER ///
    saStarts = padWidth;
    MPI_Type_create_subarray(3, globCopy.data(), loclSize.data(), saStarts.data(), MPI_ORDER_C, MPI_FP_REAL, &csSubs(0));

    saStarts = 0, 0, 0;
    MPI_Type_create_subarray(3, globCopy.data(), loclSize.data(), saStarts.data(), MPI_ORDER_C, MPI_FP_REAL, &crSubs(0));

    /// LEFT-BACK-BOTTOM CORNER ////
    saStarts = padWidth;            saStarts(1) = coreSize(1);
    MPI_Type_create_subarray(3, globCopy.data(), loclSize.data(), saStarts.data(), MPI_ORDER_C, MPI_FP_REAL, &csSubs(1));

    saStarts = 0, 0, 0;             saStarts(1) = coreSize(1) + padWidth(1);
    MPI_Type_create_subarray(3, globCopy.data(), loclSize.data(), saStarts.data(), MPI_ORDER_C, MPI_FP_REAL, &crSubs(1));

    /// RIGHT-FRONT-BOTTOM CORNER //
    saStarts = padWidth;            saStarts(0) = coreSize(0);
    MPI_Type_create_subarray(3, globCopy.data(), loclSize.data(), saStarts.data(), MPI_ORDER_C, MPI_FP_REAL, &csSubs(2));

    saStarts = 0, 0, 0;             saStarts(0) = coreSize(0) + padWidth(0);
    MPI_Type_create_subarray(3, globCopy.data(), loclSize.data(), saStarts.data(), MPI_ORDER_C, MPI_FP_REAL, &crSubs(2));

    /// RIGHT-BACK-BOTTOM CORNER ///
    saStarts = coreSize;            saStarts(2) = padWidth(2);
    MPI_Type_create_subarray(3, globCopy.data(), loclSize.data(), saStarts.data(), MPI_ORDER_C, MPI_FP_REAL, &csSubs(3));

    saStarts = coreSize + padWidth; saStarts(2) = 0;
    MPI_Type_create_subarray(3, globCopy.data(), loclSize.data(), saStarts.data(), MPI_ORDER_C, MPI_FP_REAL, &crSubs(3));


    /// LEFT-FRONT-TOP CORNER ////
    saStarts = padWidth;            saStarts(2) = coreSize(2);
    MPI_Type_create_subarray(3, globCopy.data(), loclSize.data(), saStarts.data(), MPI_ORDER_C, MPI_FP_REAL, &csSubs(4));

    saStarts = 0, 0, 0;             saStarts(2) = coreSize(2) + padWidth(2);
    MPI_Type_create_subarray(3, globCopy.data(), loclSize.data(), saStarts.data(), MPI_ORDER_C, MPI_FP_REAL, &crSubs(4));

    //// LEFT-BACK-TOP CORNER ////
    saStarts = coreSize;            saStarts(0) = padWidth(0);
    MPI_Type_create_subarray(3, globCopy.data(), loclSize.data(), saStarts.data(), MPI_ORDER_C, MPI_FP_REAL, &csSubs(5));

    saStarts = coreSize + padWidth; saStarts(0) = 0;
    MPI_Type_create_subarray(3, globCopy.data(), loclSize.data(), saStarts.data(), MPI_ORDER_C, MPI_FP_REAL, &crSubs(5));

    /// RIGHT-FRONT-TOP CORNER ////
    saStarts = coreSize;            saStarts(1) = padWidth(1);
    MPI_Type_create_subarray(3, globCopy.data(), loclSize.data(), saStarts.data(), MPI_ORDER_C, MPI_FP_REAL, &csSubs(6));

    saStarts = coreSize + padWidth; saStarts(1) = 0;
    MPI_Type_create_subarray(3, globCopy.data(), loclSize.data(), saStarts.data(), MPI_ORDER_C, MPI_FP_REAL, &crSubs(6));

    //// RIGHT-BACK-TOP CORNER ////
    saStarts = coreSize;
    MPI_Type_create_subarray(3, globCopy.data(), loclSize.data(), saStarts.data(), MPI_ORDER_C, MPI_FP_REAL, &csSubs(7));

    saStarts = coreSize + padWidth;
    MPI_Type_create_subarray(3, globCopy.data(), loclSize.data(), saStarts.data(), MPI_ORDER_C, MPI_FP_REAL, &crSubs(7));


    /////////////////////////////////// ALL SUB-ARRAYS CREATED. COMMIT THEM ///////////////////////////////////
    /** FACE SUB-ARRAYS **/
    for (int i=0; i<6; i++) {
        MPI_Type_commit(&fsSubs(i));
        MPI_Type_commit(&frSubs(i));
    }

    /** EDGE SUB-ARRAYS **/
    for (int i=0; i<12; i++) {
        MPI_Type_commit(&esSubs(i));
        MPI_Type_commit(&erSubs(i));
    }

    /** CORNER SUB-ARRAYS **/
    for (int i=0; i<8; i++) {
        MPI_Type_commit(&csSubs(i));
        MPI_Type_commit(&crSubs(i));
    }
}

/**
 ********************************************************************************************************************************************
 * \brief   Function to send data across all sub-domain faces
 *
 *          This is the core function of the mpidata class.
 *          The end slices of each sub-domain recieves data from their corresponding neighbouring sub-domains,
 *          while the interior slices of each sub-domain sends data to their corresponding neighbouring sub-domains.
 *
 *          All the data slices are send as subarray MPI derived data-types created in the \ref createSubarrays function.
 *          As a result, \ref syncData must be called only after the subarrays have been created.
 *
 *          The data transfer is implemented here with a mixture of blocking and non-blocking communication calls.
 *          The receives are non-blocking, while the sends are blocking. This combination prevents inter-processor deadlock.
 ********************************************************************************************************************************************
 */
void mpidata::syncData() {
    recvRequest = MPI_REQUEST_NULL;

    // PERFORM DATA TRANSFER ACROSS THE SIX FACES
    for (int i=0; i<6; i++)
        MPI_Irecv(dataField.dataFirst(), 1, frSubs(i), rankData.faceRanks(i), fTags(i), MPI_COMM_WORLD, &recvRequest(i));

    for (int i=0; i<6; i++)
        MPI_Send(dataField.dataFirst(), 1, fsSubs(i), rankData.faceRanks(i), i+1, MPI_COMM_WORLD);

    MPI_Waitall(6, recvRequest.dataFirst(), recvStatus.dataFirst());


    // PERFORM DATA TRANSFER ACROSS THE TWELVE EDGES
    for (int i=0; i<12; i++)
        MPI_Irecv(dataField.dataFirst(), 1, erSubs(i), rankData.edgeRanks(i), eTags(i), MPI_COMM_WORLD, &recvRequest(i));

    for (int i=0; i<12; i++)
        MPI_Send(dataField.dataFirst(), 1, esSubs(i), rankData.edgeRanks(i), i+1, MPI_COMM_WORLD);

    MPI_Waitall(12, recvRequest.dataFirst(), recvStatus.dataFirst());


    // PERFORM DATA TRANSFER ACROSS THE EIGHT CORNERS
    for (int i=0; i<8; i++)
        MPI_Irecv(dataField.dataFirst(), 1, crSubs(i), rankData.cornRanks(i), cTags(i), MPI_COMM_WORLD, &recvRequest(i));

    for (int i=0; i<8; i++)
        MPI_Send(dataField.dataFirst(), 1, csSubs(i), rankData.cornRanks(i), i+1, MPI_COMM_WORLD);

    MPI_Waitall(8, recvRequest.dataFirst(), recvStatus.dataFirst());
}
