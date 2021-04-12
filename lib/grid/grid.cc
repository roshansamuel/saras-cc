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
/*! \file grid.cc
 *
 *  \brief Definitions for functions of class grid
 *  \sa grid.h
 *  \author Roshan Samuel
 *  \date Nov 2019
 *  \copyright New BSD License
 *
 ********************************************************************************************************************************************
 */

#include "grid.h"

/**
 ********************************************************************************************************************************************
 * \brief   Constructor of the grid class
 *
 *          The class constructor initializes the mesh for computational problem.
 *          The pad widths, global grid limits in the full domain, local grid limits in the MPI decomposed sub-domains,
 *          grid spacings, domain lengths, etc., along each direction are set.
 *          Appropriate stretching functions are chosen according to user preferences and their corresponding grid
 *          transformation derivatives are also computed and stored.
 *
 * \param   solParam is a const reference to the global data contained in the parser class
 * \param   parallelData is a reference to the global data contained in the parallel class
 ********************************************************************************************************************************************
 */
grid::grid(const parser &solParam, parallel &parallelData): inputParams(solParam),
                                                            rankData(parallelData) {
    /** Depending on the finite-difference scheme chosen for calculating derivatives, set the \ref padWidths along all directions. */
    if (inputParams.dScheme == 1) {
        padWidths = 1, 1, 1;
    } else if (inputParams.dScheme == 2) {
        padWidths = 2, 2, 2;
    } else {
        if (rankData.rank == 0) {
            std::cout << "Undefined finite differencing scheme in YAML file. ABORTING" << std::endl;
        }
        MPI_Finalize();
        exit(0);
    }

    // THE ARRAY sizeArray HAS ELEMENTS [1, 3, 5, 9, 17, 33 ..... ] - STAGGERED GRID SIZE
    makeSizeArray();

    sizeIndex = inputParams.xInd, inputParams.yInd, inputParams.zInd;
    globalSize = sizeArray(sizeIndex(0)), sizeArray(sizeIndex(1)), sizeArray(sizeIndex(2));
#ifdef PLANAR
    totalPoints = globalSize(0)*globalSize(2);
#else
    totalPoints = globalSize(0)*globalSize(1)*globalSize(2);
#endif

    xLen = inputParams.Lx;
    yLen = inputParams.Ly;
    zLen = inputParams.Lz;

    thBeta = inputParams.betaX, inputParams.betaY, inputParams.betaZ;

    dXi = 1.0/real(globalSize(0) - 1);
    dEt = 1.0/real(globalSize(1) - 1);
    dZt = 1.0/real(globalSize(2) - 1);

#ifdef PLANAR
    // IS IT OKAY TO SET BELOW VALUE AS 1 EVEN WHEN inputParams.dScheme IS NOT 1?
    padWidths(1) = 1;
    yLen = 1.0;
    dEt = 1.0;
#endif

    // COMPUTE THE LOCAL ARRAY SIZES, coreSize, START AND END INDICES, subarrayStarts AND subarrayEnds
    computeGlobalLimits();

    // SET THE TinyVector AND RectDomain VARIABLES BASED ON VALUES COMPUTED IN computeGlobalLimits, FOR RESIZING ALL LOCAL GRIDS
    setDomainSizes();

    // RESIZE GRID USING THE VARIABLES CONSTRUCTED ABOVE IN setDomainSizes
    resizeGrid();

    // GENERATE THE GLOBAL TRANSFORMED GRID
    createXiEtaZeta();

    // CREATE UNIFORM GRID WHICH IS DEFAULT ALONG ALL THREE DIRECTIONS
    createUniformGrid();

    // FLAG TO CHECK FOR GRID ANISOTROPY - FALSE BY DEFAULT UNLESS NON-UNIFORM GRID IS CREATED
    bool gridCheck = false;

    // DEPENDING ON THE USER-SET PARAMETERS, SWITCH TO TAN-HYP ALONG SELECTED DIRECTIONS
    if (inputParams.xGrid == 2) {
        createTanHypGrid(0);
        gridCheck = true;
    }

    if (inputParams.yGrid == 2) {
        createTanHypGrid(1);
        gridCheck = true;
    }

    if (inputParams.zGrid == 2) {
        createTanHypGrid(2);
        gridCheck = true;
    }

    x = xGlobal(blitz::Range(subarrayStarts(0) - padWidths(0), subarrayEnds(0) + padWidths(0)));
    y = yGlobal(blitz::Range(subarrayStarts(1) - padWidths(1), subarrayEnds(1) + padWidths(1)));
    z = zGlobal(blitz::Range(subarrayStarts(2) - padWidths(2), subarrayEnds(2) + padWidths(2)));

    if (gridCheck) checkAnisotropy();
}


/**
 ********************************************************************************************************************************************
 * \brief   Function to resize and initialize the size array from which the dimensions of the grid will be determined
 *
 *          The size array will generate grid sizes according to \f$ 2^N + 2 \f$ to enable multigrid operations on the grid.
 *          The \ref parser#xInd "xInd", \ref parser#yInd "yInd" and \ref parser#zInd "zInd" parameters set by the users in
 *          parameters.yaml and read by the \ref parser class will be used to locate the grid size within the \ref sizeArray and
 *          generate grid accordingly.
 *
 *          Note that the grid sizes stored in \ref sizeArray correspond to the collocated grid.
 *          For multi-grid operations, the number of grid points necessary is \f$ 2^N + 1 \f$, which is 1 less than the values
 *          generated by this function.
 *          However, multi-grid is applied here to compute pressure correction and pressure is calculated on the staggered grid.
 *          Since there are \f$ N - 1 \f$ staggered grid points for \f$ N \f$ collocated points, the sizes become consistent.
 ********************************************************************************************************************************************
 */
void grid::makeSizeArray() {
    int maxIndex = 15;

    sizeArray.resize(maxIndex);
    for (int i=0; i < maxIndex; i++) {
        sizeArray(i) = int(pow(2, i)) + 1;
    }

    sizeArray(0) = 1;
}


/**
 ********************************************************************************************************************************************
 * \brief   Function to set the extent of local sub-domains in terms of the global index of the full domain
 *
 *          Depending on the number of processor divisions along each direction, the limits of the grid for each local
 *          sub-domain is set based on its \ref parallel#xRank "xRank" and \ref parallel#yRank "yRank".
 *          These limits are used to locate the local sub-domains within the full domain later.
 ********************************************************************************************************************************************
 */
void grid::computeGlobalLimits() {
    int xiSt, etSt, ztSt;
    int xiEn, etEn, ztEn;
    int localNx, localNy, localNz;

    // NUMBER OF STAGGERED POINTS IN EACH SUB-DOMAIN EXCLUDING PAD POINTS
    localNx = (globalSize(0) - 1)/rankData.npX + 1;
#ifndef PLANAR
    localNy = (globalSize(1) - 1)/rankData.npY + 1;
#else
    localNy = 1;
#endif
    localNz = (globalSize(2));

    // SETTING GLOBAL LIMITS
    // ADD ONE EXTRA POINT EACH AT FIRST AND LAST SUB-DOMAINS
    // FIRST SET THE LIMITS TO DEFAULT VALUES - THIS ELIMINATES AN EXTRA 'if' CONDITION
    // THEN SET LIMITS FOR LAST RANK IN EACH DIRECTION FIRST AND *FINALLY* SET LIMITS OF 0TH RANK
    // THIS IS NECESSARY TO AVOID ERRORS WHEN A PROCESSOR IS BOTH FIRST AND LAST RANK
    // THIS HAPPENS WHEN THERE ARE NO DIVISIONS ALONG AN AXIS AS ALONG Z-DIRECTION

    // ALONG XI-DIRECTION
    xiSt = rankData.xRank*(localNx - 1);
    xiEn = xiSt + localNx - 1;

    // ALONG ETA-DIRECTION
    etSt = rankData.yRank*(localNy - 1);
    etEn = etSt + localNy - 1;

    // ALONG ZETA-DIRECTION
    ztSt = 0;
    ztEn = ztSt + localNz - 1;

    coreSize = localNx, localNy, localNz;
    fullSize = coreSize + 2*padWidths;

    // SUB-ARRAY STARTS AND ENDS FOR *STAGGERED* GRID
    subarrayStarts = xiSt, etSt, ztSt;
    subarrayEnds = xiEn, etEn, ztEn;
}


/**
 ********************************************************************************************************************************************
 * \brief   Function to set all the TinyVector and RectDomain variables for all future references throughout the solver
 *
 *          The function sets the core and full domain sizes for all the sub-domains after MPI decomposition.
 *          Additionally, the pad widths and starting indices of the sub-domains within the global domain are also set.
 ********************************************************************************************************************************************
 */
void grid::setDomainSizes() {
    blitz::TinyVector<int, 3> loBound, upBound;

    // LOWER BOUND AND UPPER BOUND OF CORE - USED TO CONSTRUCT THE CORE SLICE OF STAGGERED POINTS
    loBound = 0, 0, 0;
    upBound = coreSize - 1;
    coreDomain = blitz::RectDomain<3>(loBound, upBound);

    // LOWER BOUND AND UPPER BOUND OF FULL SUB-DOMAIN - USED TO CONSTRUCT THE FULL SUB-DOMAIN SLICE
    loBound = -padWidths;
    upBound = coreSize + padWidths - 1;
    fullDomain = blitz::RectDomain<3>(loBound, upBound);
}


/**
 ********************************************************************************************************************************************
 * \brief   Function to resize and initialize the grid
 *
 *          The global collocated grid in transformed plane are resized according the global size of full domain.
 *          Then the local collocated grid in transformed plane is resized according to the limits defined in \ref computeGlobalLimits.
 *          Correspondingly, this function is called after the global limits have been set.
 *          After defining the transformed plane coordinates, both the staggered and collocated grids in physical plane are resized.
 *          Finally, the arrays for the grid derivative terms are also resized and initialized to 1.
 ********************************************************************************************************************************************
 */
void grid::resizeGrid() {
    // ALL ARRAYS MUST BE RESIZED AND REINDEXED TO LET THE NEGATIVE PADS HAVE NEGATIVE INDICES
    // THIS IS DONE IN A SINGLE STEP BY INITIALIZING THE ARRAYS WITH A blitz::Range OBJECT WHICH CONTAINS SIZE AND INDEXING INFORMATION
    blitz::Range xRange, yRange, zRange;

    // FIRST SET RANGES TO RESIZE GLOBAL ARRAYS
    xRange = blitz::Range(-padWidths(0), globalSize(0));
    yRange = blitz::Range(-padWidths(1), globalSize(1));
    zRange = blitz::Range(-padWidths(2), globalSize(2));

    // LOCAL GRID POINTS AND THEIR METRICS
    xGlobal.resize(xRange);
    yGlobal.resize(yRange);
    zGlobal.resize(zRange);

    // RANGE OF THE SUB-DOMAIN FOR STAGGERED AND COLLOCATED GRIDS: CONSTRUCTED FROM LOWER AND UPPER BOUNDS OF FULL SUB-DOMAIN
    xRange = blitz::Range(fullDomain.lbound(0), fullDomain.ubound(0));
    yRange = blitz::Range(fullDomain.lbound(1), fullDomain.ubound(1));
    zRange = blitz::Range(fullDomain.lbound(2), fullDomain.ubound(2));

    // LOCAL XI, ETA AND ZETA ARRAYS
    xi.resize(xRange);
    et.resize(yRange);
    zt.resize(zRange);

    // LOCAL GRID POINTS AND THEIR METRICS
    x.resize(xRange);
    y.resize(yRange);
    z.resize(zRange);

    xi_x.resize(xRange);        xixx.resize(xRange);        xix2.resize(xRange);
    et_y.resize(yRange);        etyy.resize(yRange);        ety2.resize(yRange);
    zt_z.resize(zRange);        ztzz.resize(zRange);        ztz2.resize(zRange);

    x = 1.0;        y = 1.0;        z = 1.0;

    // BELOW ARE DEFAULT VALUES FOR A UNIFORM GRID OVER DOMAIN OF LENGTH 1.0
    // THESE VALUES ARE OVERWRITTEN AS PER GRID TYPE
    xi_x = 1.0;     xixx = 0.0;     xix2 = 1.0;
    et_y = 1.0;     etyy = 0.0;     ety2 = 1.0;
    zt_z = 1.0;     ztzz = 0.0;     ztz2 = 1.0;
}


/**
 ********************************************************************************************************************************************
 * \brief   Function to compute global values of xi, eta and zeta in transformed plane
 *
 *          The function populates the \ref xiGlo, \ref etGlo and \ref ztGlo arrays from which the local values of
 *          \ref xi, \ref et and \ref zt in each sub-domain are obtained.
 *          These local values are obtained from the global grid according to the limits defined in \ref computeGlobalLimits.
 ********************************************************************************************************************************************
 */
void grid::createXiEtaZeta() {
    int i;

    xiGlo.resize(globalSize(0) + 2*padWidths(0));          xiGlo.reindexSelf(-padWidths(0));
    etGlo.resize(globalSize(1) + 2*padWidths(1));          etGlo.reindexSelf(-padWidths(1));
    ztGlo.resize(globalSize(2) + 2*padWidths(2));          ztGlo.reindexSelf(-padWidths(2));

    // ALONG XI-DIRECTION
    for (i = -padWidths(0); i < globalSize(0) + padWidths(0); i++) {
        xiGlo(i) = real(i)*dXi;
    }

    // ALONG ETA-DIRECTION
    for (i = -padWidths(1); i < globalSize(1) + padWidths(1); i++) {
        etGlo(i) = real(i)*dEt;
    }

    // ALONG ZETA-DIRECTION
    for (i = -padWidths(2); i < globalSize(2) + padWidths(2); i++) {
        ztGlo(i) = real(i)*dZt;
    }

    // SET LOCAL TRANSFORMED GRID AS SLICES FROM THE GLOBAL TRANSFORMED GRID GENERATED ABOVE
    xi = xiGlo(blitz::Range(subarrayStarts(0) - padWidths(0), subarrayEnds(0) + padWidths(0)));
    et = etGlo(blitz::Range(subarrayStarts(1) - padWidths(1), subarrayEnds(1) + padWidths(1)));
    zt = ztGlo(blitz::Range(subarrayStarts(2) - padWidths(2), subarrayEnds(2) + padWidths(2)));
}


/**
 ********************************************************************************************************************************************
 * \brief   Function to generate grid with uniform stretching
 *
 *          The local collocated grids, \ref xColloc, \ref yColloc, and \ref zColloc are equated to their corresponding
 *          transformed plane coordinates, \ref xi, \ref et, and \ref zt respectively.
 *          The corresponding grid derivative terms, \ref xi_xColloc, \ref xixxColloc, \ref ety2Colloc, etc are left
 *          unchanged from their initial value of 1.0, indicating that the grid is uniform.
 *
 *          Similarly, the staggered grids, \ref xStaggr, \ref yStaggr, and \ref zStaggr are also equated to the mid-point
 *          averaged values of the nodes in their corresponding transformed plane coordinates, \ref xi, \ref et, and \ref zt
 *          respectively.
 *          As before, the grid derivative terms for the staggered points are also left as 1.0.
 ********************************************************************************************************************************************
 */
void grid::createUniformGrid() {
    int i;

    // COLLOCATED X-GRID POINTS FROM UNIFORM XI-GRID POINTS AND THEIR METRICS
    for (i = -padWidths(0); i < globalSize(0) + padWidths(0); i++) {
        xGlobal(i) = xLen*xiGlo(i);
    }

#ifndef PLANAR
    // COLLOCATED Y-GRID POINTS FROM UNIFORM ETA-GRID POINTS AND THEIR METRICS
    for (i = -padWidths(1); i < globalSize(1) + padWidths(1); i++) {
        yGlobal(i) = yLen*etGlo(i);
    }
#endif

    // COLLOCATED Z-GRID POINTS FROM UNIFORM ZETA-GRID POINTS AND THEIR METRICS
    for (i = -padWidths(2); i < globalSize(2) + padWidths(2); i++) {
        zGlobal(i) = zLen*ztGlo(i);
    }

    xi_x = 1.0/xLen;
    xix2 = pow(xi_x, 2.0);

    et_y = 1.0/yLen;
    ety2 = pow(et_y, 2.0);

    zt_z = 1.0/zLen;
    ztz2 = pow(zt_z, 2.0);
}


/**
 ********************************************************************************************************************************************
 * \brief   Function to generate grid with tangent-hyperbolic stretching
 *
 *          The local collocated grids, \ref xColloc, \ref yColloc, and \ref zColloc are initialized from their corresponding
 *          transformed plane coordinates, \ref xi, \ref et, and \ref zt respectively using the tangent hyperbolic function.
 *          The corresponding grid derivative terms, \ref xi_xColloc, \ref xixxColloc, \ref ety2Colloc, etc are computed
 *          using analytical expressions for the tangent hyperbolic function.
 *
 *          Similarly, the staggered grids, \ref xStaggr, \ref yStaggr, and \ref zStaggr are initialized from the mid-point
 *          averaged values of the nodes in their corresponding transformed plane coordinates, \ref xi, \ref et, and \ref zt
 *          respectively using the tangent hyperbolic function.
 *          As before, the grid derivative terms for the staggered points are also computed using analytical expressions.
 *
 * \param   dim is an integer value that defines the direction along which tan-hyp grid is to be generated: 0 -> X, 1 -> Y, 2 -> Z
 ********************************************************************************************************************************************
 */
void grid::createTanHypGrid(int dim) {
    int i;
    blitz::Array<real, 1> df_x, dfxx, dfx2;
    blitz::Range lftPts, rgtPts;

#ifndef TEST_RUN
    if (rankData.rank == 0) {
        switch (dim) {
            case 0: std::cout << "Generating tangent hyperbolic grid along X direction" << std::endl;
                    break;
            case 1: std::cout << "Generating tangent hyperbolic grid along Y direction" << std::endl;
                    break;
            case 2: std::cout << "Generating tangent hyperbolic grid along Z direction" << std::endl;
                    break;
        }
    }
#endif

    if (dim == 0) {
        // GENERATE X-GRID POINTS FROM UNIFORM XI-GRID POINTS AND THEIR METRICS
        df_x.resize(blitz::Range(-padWidths(0), globalSize(0)));
        dfxx.resize(blitz::Range(-padWidths(0), globalSize(0)));
        dfx2.resize(blitz::Range(-padWidths(0), globalSize(0)));

        for (i = 0; i < globalSize(0); i++) {
            xGlobal(i) = xLen*(1.0 - tanh(thBeta[0]*(1.0 - 2.0*xiGlo(i)))/tanh(thBeta[0]))/2.0;

            df_x(i) = tanh(thBeta[0])/(thBeta[0]*xLen*(1.0 - pow((1.0 - 2.0*x(i)/xLen)*tanh(thBeta[0]), 2)));
            dfxx(i) = -4.0*pow(tanh(thBeta[0]), 3)*(1.0 - 2.0*x(i)/xLen)/(thBeta[0]*xLen*xLen*pow(1.0 - pow(tanh(thBeta[0])*(1.0 - 2.0*x(i)/xLen), 2), 2));
            dfx2(i) = pow(df_x(i), 2.0);
        }

        lftPts = blitz::Range(-padWidths(0), -1, 1);
        rgtPts = blitz::Range(globalSize(0) - padWidths(0) - 1, globalSize(0) - 2, 1);

        xGlobal(lftPts) = xGlobal(rgtPts) - xLen;
        df_x(lftPts) = df_x(rgtPts);
        dfxx(lftPts) = dfxx(rgtPts);
        dfx2(lftPts) = dfx2(rgtPts);

        rgtPts = blitz::Range(globalSize(0), globalSize(0) + padWidths(0) - 1, 1);
        lftPts = blitz::Range(1, padWidths(0), 1);

        xGlobal(rgtPts) = xLen + xGlobal(lftPts);
        df_x(rgtPts) = df_x(lftPts);
        dfxx(rgtPts) = dfxx(lftPts);
        dfx2(rgtPts) = dfx2(lftPts);

        xi_x = df_x(blitz::Range(subarrayStarts(0) - padWidths(0), subarrayEnds(0) + padWidths(0)));
        xixx = dfxx(blitz::Range(subarrayStarts(0) - padWidths(0), subarrayEnds(0) + padWidths(0)));
        xix2 = dfx2(blitz::Range(subarrayStarts(0) - padWidths(0), subarrayEnds(0) + padWidths(0)));
    }

#ifndef PLANAR
    if (dim == 1) {
        // STAGGERED Y-GRID POINTS FROM UNIFORM ETA-GRID POINTS AND THEIR METRICS
        df_x.resize(blitz::Range(-padWidths(1), globalSize(1)));
        dfxx.resize(blitz::Range(-padWidths(1), globalSize(1)));
        dfx2.resize(blitz::Range(-padWidths(1), globalSize(1)));

        for (i = 0; i < globalSize(1); i++) {
            yGlobal(i) = yLen*(1.0 - tanh(thBeta[1]*(1.0 - 2.0*etGlo(i)))/tanh(thBeta[1]))/2.0;

            df_x(i) = tanh(thBeta[1])/(thBeta[1]*yLen*(1.0 - pow((1.0 - 2.0*y(i)/yLen)*tanh(thBeta[1]), 2)));
            dfxx(i) = -4.0*pow(tanh(thBeta[1]), 3)*(1.0 - 2.0*y(i)/yLen)/(thBeta[1]*yLen*yLen*pow(1.0 - pow(tanh(thBeta[1])*(1.0 - 2.0*y(i)/yLen), 2), 2));
            dfx2(i) = pow(df_x(i), 2.0);
        }

        lftPts = blitz::Range(-padWidths(1), -1, 1);
        rgtPts = blitz::Range(globalSize(1) - padWidths(1) - 1, globalSize(1) - 2, 1);

        yGlobal(lftPts) = yGlobal(rgtPts) - yLen;
        df_x(lftPts) = df_x(rgtPts);
        dfxx(lftPts) = dfxx(rgtPts);
        dfx2(lftPts) = dfx2(rgtPts);

        rgtPts = blitz::Range(globalSize(1), globalSize(1) + padWidths(1) - 1, 1);
        lftPts = blitz::Range(1, padWidths(1), 1);

        yGlobal(rgtPts) = yLen + yGlobal(lftPts);
        df_x(rgtPts) = df_x(lftPts);
        dfxx(rgtPts) = dfxx(lftPts);
        dfx2(rgtPts) = dfx2(lftPts);

        et_y = df_x(blitz::Range(subarrayStarts(1) - padWidths(1), subarrayEnds(1) + padWidths(1)));
        etyy = dfxx(blitz::Range(subarrayStarts(1) - padWidths(1), subarrayEnds(1) + padWidths(1)));
        ety2 = dfx2(blitz::Range(subarrayStarts(1) - padWidths(1), subarrayEnds(1) + padWidths(1)));
    }
#endif

    if (dim == 2) {
        // STAGGERED Z-GRID POINTS FROM UNIFORM ZETA-GRID POINTS AND THEIR METRICS
        df_x.resize(blitz::Range(-padWidths(2), globalSize(2)));
        dfxx.resize(blitz::Range(-padWidths(2), globalSize(2)));
        dfx2.resize(blitz::Range(-padWidths(2), globalSize(2)));

        for (i = 0; i < globalSize(2); i++) {
            zGlobal(i) = zLen*(1.0 - tanh(thBeta[2]*(1.0 - 2.0*ztGlo(i)))/tanh(thBeta[2]))/2.0;

            df_x(i) = tanh(thBeta[2])/(thBeta[2]*zLen*(1.0 - pow((1.0 - 2.0*z(i)/zLen)*tanh(thBeta[2]), 2)));
            dfxx(i) = -4.0*pow(tanh(thBeta[2]), 3)*(1.0 - 2.0*z(i)/zLen)/(thBeta[2]*zLen*zLen*pow(1.0 - pow(tanh(thBeta[2])*(1.0 - 2.0*z(i)/zLen), 2), 2));
            dfx2(i) = pow(df_x(i), 2.0);
        }

        lftPts = blitz::Range(-padWidths(2), -1, 1);
        rgtPts = blitz::Range(globalSize(2) - padWidths(2) - 1, globalSize(2) - 2, 1);

        zGlobal(lftPts) = zGlobal(rgtPts) - zLen;
        df_x(lftPts) = df_x(rgtPts);
        dfxx(lftPts) = dfxx(rgtPts);
        dfx2(lftPts) = dfx2(rgtPts);

        rgtPts = blitz::Range(globalSize(2), globalSize(2) + padWidths(2) - 1, 1);
        lftPts = blitz::Range(1, padWidths(2), 1);

        zGlobal(rgtPts) = zLen + zGlobal(lftPts);
        df_x(rgtPts) = df_x(lftPts);
        dfxx(rgtPts) = dfxx(lftPts);
        dfx2(rgtPts) = dfx2(lftPts);

        zt_z = df_x(blitz::Range(subarrayStarts(2) - padWidths(2), subarrayEnds(2) + padWidths(2)));
        ztzz = dfxx(blitz::Range(subarrayStarts(2) - padWidths(2), subarrayEnds(2) + padWidths(2)));
        ztz2 = dfx2(blitz::Range(subarrayStarts(2) - padWidths(2), subarrayEnds(2) + padWidths(2)));
    }
}


/**
 ********************************************************************************************************************************************
 * \brief   Function to check the anisotropy of the grid
 *
 *          The function is called only if non-uniform grid is made.
 *          It scans through the entire grid cell-by-cell and checks the 2 aspect-ratios of the cell (one for 2D grids)
 *          The maximum value of aspect ratio is stored.
 *          Each MPI sub-domain checks within its limits and an MPI_Reduce call gets the global maximum.
 *
 ********************************************************************************************************************************************
 */
void grid::checkAnisotropy() {
    real cellMaxAR;
    real xWidth, zWidth;
    real localMax, globalMax;

#ifndef PLANAR
    real yWidth;
    real aRatio, bRatio;
    real xyRatio, yzRatio, zxRatio;
#endif

    localMax = 0.0;
#ifdef PLANAR
    for (int i = 0; i <= coreDomain.ubound(0) + 1; i++) {
        for (int k = 0; k <= coreDomain.ubound(2) + 1; k++) {
            xWidth = x(i-1) - x(i);
            zWidth = z(k-1) - z(k);
            cellMaxAR = std::max(xWidth/zWidth, zWidth/xWidth);
            if (cellMaxAR > localMax) localMax = cellMaxAR;

        }
    }
#else
    for (int i = 0; i <= coreDomain.ubound(0) + 1; i++) {
        for (int j = 0; j <= coreDomain.ubound(1) + 1; j++) {
            for (int k = 0; k <= coreDomain.ubound(2) + 1; k++) {
                xWidth = x(i-1) - x(i);
                yWidth = y(j-1) - y(j);
                zWidth = z(k-1) - z(k);
                xyRatio = std::max(xWidth/yWidth, yWidth/xWidth);
                yzRatio = std::max(yWidth/zWidth, zWidth/yWidth);
                zxRatio = std::max(zWidth/xWidth, xWidth/zWidth);
                aRatio = std::max(xyRatio, yzRatio);
                bRatio = std::max(yzRatio, zxRatio);
                cellMaxAR = std::max(aRatio, bRatio);
                if (cellMaxAR > localMax) localMax = cellMaxAR;
            }
        }
    }
#endif

    MPI_Allreduce(&localMax, &globalMax, 1, MPI_FP_REAL, MPI_MAX, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    if (globalMax > 7.0) {
        if (rankData.rank == 0) std::cout << "\nWARNING: Grid anisotropy exceeds limits. Finite-difference calculations will be inaccurate" << std::endl;
    } else {
        if (rankData.rank == 0) std::cout << "\nMaximum grid anisotropy is " << globalMax << std::endl;
    }
}

