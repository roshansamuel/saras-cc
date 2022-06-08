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
/*! \file main.cc
 *
 *  \brief Main file for post-processing run of Saras.
 *
 *  \author Roshan Samuel
 *  \date Nov 2022
 *  \copyright New BSD License
 *
 ********************************************************************************************************************************************
 */

#include "global.h"
#include "postprocess.h"

int main() {
    int mpiThreadProvided;
    struct timeval runStart, runEnd;

    // INITIALIZE MPI
    MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &mpiThreadProvided);

    // ALL PROCESSES READ THE INPUT PARAMETERS
    parser inputParams;

    // INITIALIZE PARALLELIZATION DATA
    parallel mpi(inputParams);

    // WRITE CONTENTS OF THE INPUT YAML FILE TO THE STANDARD I/O
    //if (mpi.rank == 0) {
    //    inputParams.writeParams();
    //}

    // INITIALIZE GRID DATA
    grid gridData(inputParams, mpi);

    // INITIALIZE POST-PROCESSING GLOBALS
    global gloData(gridData);

    // ENABLE-DISABLE PERIODIC DATA TRANSFER IN GLOBALS
    gloData.checkPeriodic(inputParams, mpi);

    if (mpiThreadProvided < MPI_THREAD_MULTIPLE)
        if (gridData.pf)
            std::cout << "\nWARNING: MPI does not provide desired threading level" << std::endl;

    gettimeofday(&runStart, NULL);

    std::vector<real> timeList = inputParams.readTimes();

    dissipation(gloData, timeList);

    gettimeofday(&runEnd, NULL);
    real run_time = ((runEnd.tv_sec - runStart.tv_sec)*1000000u + runEnd.tv_usec - runStart.tv_usec)/1.e6;

    if (mpi.rank == 0) {
        std::cout << std::endl << "Post-processing completed" << std::endl;
        std::cout << std::endl;
        std::cout << "Time taken for processing: " << run_time << std::endl;
    }

    // FINALIZE AND CLEAN-UP
    MPI_Finalize();

    return 0;
}
