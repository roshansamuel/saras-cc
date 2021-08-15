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
/*! \file parallel.cc
 *
 *  \brief Definitions for functions of class parallel
 *  \sa parallel.h
 *  \author Roshan Samuel
 *  \date Nov 2019
 *  \copyright New BSD License
 *
 ********************************************************************************************************************************************
 */

#include "parallel.h"

/**
 ********************************************************************************************************************************************
 * \brief   Constructor of the parallel class
 *
 *          The initializing functions of MPI are called in order to get the total number of processes spawned, and
 *          the rank of each process.
 *          The xRank and yRank of each process are calculated and assigned.
 *          Finally, the ranks of neighbouring processes are found and stored in an array for use in MPI communications
 *
 * \param   iDat is a const reference to the global data contained in the parser class
 ********************************************************************************************************************************************
 */
parallel::parallel(const parser &iDat): npX(iDat.npX), npY(iDat.npY), npZ(iDat.npZ) {
    // GET EACH PROCESSES' RANK AND TOTAL NUMBER OF PROCESSES
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nProc);

    // ABORT IF THE NUMBER OF PROCESSORS IN EACH DIRECTION SPECIFIED IN INPUT DOES NOT MATCH WITH AVAILABLE CORES
    if (npX*npY*npZ != nProc) {
        if (rank == 0) {
            std::cout << "ERROR: Number of processors specified in input file does not match. Aborting" << std::endl;
        }
        MPI_Finalize();
        exit(0);
    }

    // ASSIGN EACH PROCESSES' xRank AND yRank
    assignRanks();

    // GET AND STORE THE RANKS OF ALL NEIGHBOURING PROCESSES FOR FUTURE DATA TRANSFER
    getNeighbours();
}

/**
 ********************************************************************************************************************************************
 * \brief   Function to assign the xRank and yRank for each sub-domain according to their global rank
 *
 *          It uses the number of sub-divisions prescribed in each direction, i.e. \ref npX and \ref npY to calculate the
 *          xRank and yRank appropriately.
 ********************************************************************************************************************************************
 */
inline void parallel::assignRanks() {
    xRank = rank % npX;
    yRank = (rank % (npX*npY)) / npX;
    zRank = rank / (npX*npY);
}

/**
 ********************************************************************************************************************************************
 * \brief   Function to get the ranks of each neighbouring sub-domain which shares a face with the given sub-domain
 *
 *          Since the solver uses pencil decomposition, it locates the ranks of a maximum of 4 neighbouring sub-domains.
 ********************************************************************************************************************************************
 */
void parallel::getNeighbours() {
    // EACH PROCESS HAS 6 NEIGHBOURS CORRESPONDING TO THE 6 FACES OF EACH CUBOIDAL SUB-DOMAIN
    faceRanks.resize(6);

    // EACH PROCESS ALSO HAS 12 EDGE NEIGHBOURS CORRESPONDING TO THE 12 EDGES OF EACH CUBOIDAL SUB-DOMAIN
    edgeRanks.resize(12);

    // EACH PROCESS ALSO HAS 8 CORNER NEIGHBOURS CORRESPONDING TO THE 8 CORNER OF EACH CUBOIDAL SUB-DOMAIN
    cornRanks.resize(8);

    // EACH PROCESS IS ASSUMED TO HAVE NO NEIGHBOURS INITIALLY
    faceRanks = MPI_PROC_NULL;
    edgeRanks = MPI_PROC_NULL;
    cornRanks = MPI_PROC_NULL;

    // INITIAL FACE NEIGHBOUR ASSIGNMENTS ARE DONE ASSUMING PERIODIC DOMAIN
    // ALONG X/XI DIRECTION
    faceRanks(0) = findRank(xRank - 1, yRank, zRank);
    faceRanks(1) = findRank(xRank + 1, yRank, zRank);

    // ALONG Y/ETA DIRECTION
#ifndef PLANAR
    faceRanks(2) = findRank(xRank, yRank - 1, zRank);
    faceRanks(3) = findRank(xRank, yRank + 1, zRank);
#endif

    // ALONG Z/ZETA DIRECTION
    faceRanks(4) = findRank(xRank, yRank, zRank - 1);
    faceRanks(5) = findRank(xRank, yRank, zRank + 1);

    // PROCESS HAS EDGE NEIGHBOURS ONLY IN 3D SIMULATIONS
#ifndef PLANAR
    edgeRanks(0)  = findRank(xRank - 1, yRank - 1, zRank);
    edgeRanks(1)  = findRank(xRank - 1, yRank + 1, zRank);

    edgeRanks(2)  = findRank(xRank + 1, yRank - 1, zRank);
    edgeRanks(3)  = findRank(xRank + 1, yRank + 1, zRank);

    edgeRanks(4)  = findRank(xRank, yRank - 1, zRank - 1);
    edgeRanks(5)  = findRank(xRank, yRank - 1, zRank + 1);

    edgeRanks(6)  = findRank(xRank, yRank + 1, zRank - 1);
    edgeRanks(7)  = findRank(xRank, yRank + 1, zRank + 1);

    edgeRanks(8)  = findRank(xRank - 1, yRank, zRank - 1);
    edgeRanks(9)  = findRank(xRank + 1, yRank, zRank - 1);

    edgeRanks(10) = findRank(xRank - 1, yRank, zRank + 1);
    edgeRanks(11) = findRank(xRank + 1, yRank, zRank + 1);
#endif
                                                        
    // PROCESS HAS CORNER NEIGHBOURS ONLY IN 3D SIMULATIONS
#ifndef PLANAR
    cornRanks(0)  = findRank(xRank - 1, yRank - 1, zRank - 1);
    cornRanks(1)  = findRank(xRank - 1, yRank + 1, zRank - 1);
    cornRanks(2)  = findRank(xRank + 1, yRank - 1, zRank - 1);
    cornRanks(3)  = findRank(xRank + 1, yRank + 1, zRank - 1);

    cornRanks(4)  = findRank(xRank - 1, yRank - 1, zRank + 1);
    cornRanks(5)  = findRank(xRank - 1, yRank + 1, zRank + 1);
    cornRanks(6)  = findRank(xRank + 1, yRank - 1, zRank + 1);
    cornRanks(7)  = findRank(xRank + 1, yRank + 1, zRank + 1);
#endif
}

