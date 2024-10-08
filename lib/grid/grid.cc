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
#include <vector>
#include <string>

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
    rankData(parallelData), xLen(inputParams.Lx), yLen(inputParams.Ly), zLen(inputParams.Lz)
{
    // Flag to enable printing to I/O only by 0 rank
    pf = false;
    if (rankData.rank == 0) pf = true;

    /** Depending on the finite-difference scheme chosen for calculating derivatives, set the \ref padWidths along all directions. */
    if (inputParams.dScheme == 1) {
        padWidths = 1, 1, 1;
    } else if (inputParams.dScheme == 2) {
        padWidths = 2, 2, 2;
    } else {
        if (pf) std::cout << "Undefined finite differencing scheme in YAML file. ABORTING" << std::endl;
        MPI_Finalize();
        exit(0);
    }

    globalSize = inputParams.Nx, inputParams.Ny, inputParams.Nz;
#ifdef PLANAR
    totalPoints = globalSize(0)*globalSize(2);
#else
    totalPoints = globalSize(0)*globalSize(1)*globalSize(2);
#endif

    thBeta = inputParams.betaX, inputParams.betaY, inputParams.betaZ;

    dXi = 1.0/real(globalSize(0));
    dEt = 1.0/real(globalSize(1));
    dZt = 1.0/real(globalSize(2));

#ifdef PLANAR
    // IS IT OKAY TO SET BELOW VALUE AS 1 EVEN WHEN inputParams.dScheme IS NOT 1?
    padWidths(1) = 1;
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

    // INITIALIZE THE GRID METRICS FOR DIFFERENT LEVELS OF MULTI-GRID V-CYCLE
    mgGridMetrics();

    // FLAG TO CHECK FOR GRID ANISOTROPY - FALSE BY DEFAULT UNLESS NON-UNIFORM GRID IS CREATED
    bool gridCheck = false;

    // DEPENDING ON THE USER-SET PARAMETERS, SWITCH TO TAN-HYP ALONG SELECTED DIRECTIONS
    if (inputParams.xyzGrid[0] > 0) {
        createTanHypGrid(0, xGlobal, xiGlo);
        gridCheck = true;
    }

#ifndef PLANAR
    if (inputParams.xyzGrid[1] > 0) {
        createTanHypGrid(1, yGlobal, etGlo);
        gridCheck = true;
    }
#endif

    if (inputParams.xyzGrid[2] > 0) {
        createTanHypGrid(2, zGlobal, ztGlo);
        gridCheck = true;
    }

#ifndef POST_RUN
    x = xGlobal(blitz::Range(subarrayStarts(0) - padWidths(0), subarrayEnds(0) + padWidths(0)));
    y = yGlobal(blitz::Range(subarrayStarts(1) - padWidths(1), subarrayEnds(1) + padWidths(1)));
    z = zGlobal(blitz::Range(subarrayStarts(2) - padWidths(2), subarrayEnds(2) + padWidths(2)));

    if (gridCheck) checkAnisotropy();
#else
    // For post-processing, X, Y and Z are defined differently to perform
    // volume integrations correctly. This doesn't affect derivative calculation.
    x = xGlobal(blitz::Range(subarrayStarts(0) - 1, subarrayEnds(0) + 1));
    y = yGlobal(blitz::Range(subarrayStarts(1) - 1, subarrayEnds(1) + 1));
    z = zGlobal(blitz::Range(subarrayStarts(2) - 1, subarrayEnds(2) + 1));

    if (rankData.xRank == 0) x(-1) = 0;
    if (rankData.xRank == rankData.npX - 1) x(coreDomain.ubound(0) + 1) = xLen;

    if (rankData.yRank == 0) y(-1) = 0;
    if (rankData.yRank == rankData.npY - 1) y(coreDomain.ubound(1) + 1) = yLen;

    if (rankData.zRank == 0) z(-1) = 0;
    if (rankData.zRank == rankData.npZ - 1) z(coreDomain.ubound(2) + 1) = zLen;
#endif
}


/**
 ********************************************************************************************************************************************
 * \brief   Function to set the extent of local sub-domains in terms of the global index of the full domain
 *
 *          Depending on the number of processor divisions along each direction, the limits of the grid for each local
 *          sub-domain is set based on its \ref parallel#xRank "xRank", \ref parallel#yRank "yRank" and \ref parallel#zRank "zRank".
 *          These limits are used to locate the local sub-domains within the full domain later.
 ********************************************************************************************************************************************
 */
void grid::computeGlobalLimits() {
    int xiSt, etSt, ztSt;
    int xiEn, etEn, ztEn;
    int localNx, localNy, localNz;
    blitz::TinyVector<int, 3> procList;

    procList = rankData.npX, rankData.npY, rankData.npZ;
    for (int i=0; i<3; i++) {
        if (globalSize(i)%procList(i) != 0) {
            if (pf) std::cout << "ERROR: Given grid sizes and processor counts do not divide evenly across MPI sub-domains. Aborting" << std::endl;
            MPI_Finalize();
            exit(0);
        }
    }

    // NUMBER OF CELLS IN EACH SUB-DOMAIN EXCLUDING GHOST CELLS
    localNx = globalSize(0)/rankData.npX;
    localNy = globalSize(1)/rankData.npY;
    localNz = globalSize(2)/rankData.npZ;

#ifdef PLANAR
    localNy = 1;
#endif

    // SETTING GLOBAL LIMITS
    // ADD ONE EXTRA POINT EACH AT FIRST AND LAST SUB-DOMAINS
    // FIRST SET THE LIMITS TO DEFAULT VALUES - THIS ELIMINATES AN EXTRA 'if' CONDITION
    // THEN SET LIMITS FOR LAST RANK IN EACH DIRECTION FIRST AND *FINALLY* SET LIMITS OF 0TH RANK
    // THIS IS NECESSARY TO AVOID ERRORS WHEN A PROCESSOR IS BOTH FIRST AND LAST RANK
    // THIS HAPPENS WHEN THERE ARE NO DIVISIONS ALONG AN AXIS AS ALONG Z-DIRECTION

    // ALONG XI-DIRECTION
    xiSt = rankData.xRank*localNx;
    xiEn = xiSt + localNx - 1;

    // ALONG ETA-DIRECTION
    etSt = rankData.yRank*localNy;
    etEn = etSt + localNy - 1;

    // ALONG ZETA-DIRECTION
    ztSt = rankData.zRank*localNz;
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
 *          Finally, the arrays for the grid derivative terms are also resized.
 ********************************************************************************************************************************************
 */
void grid::resizeGrid() {
    // ALL ARRAYS MUST BE RESIZED AND REINDEXED TO LET THE NEGATIVE PADS HAVE NEGATIVE INDICES
    // THIS IS DONE IN A SINGLE STEP BY INITIALIZING THE ARRAYS WITH A blitz::Range OBJECT WHICH CONTAINS SIZE AND INDEXING INFORMATION
    blitz::Range xRange, yRange, zRange;

    // FIRST SET RANGES TO RESIZE GLOBAL ARRAYS
    xRange = blitz::Range(-padWidths(0), globalSize(0) + padWidths(0) - 1);
    yRange = blitz::Range(-padWidths(1), globalSize(1) + padWidths(1) - 1);
    zRange = blitz::Range(-padWidths(2), globalSize(2) + padWidths(2) - 1);

    // GLOBAL XI, ETA AND ZETA ARRAYS
    xiGlo.resize(xRange);
    etGlo.resize(yRange);
    ztGlo.resize(zRange);

    // GLOBAL GRID POINTS AND THEIR METRICS
    xGlobal.resize(xRange);
    yGlobal.resize(yRange);
    zGlobal.resize(zRange);

    // RANGE OF THE SUB-DOMAIN FOR LOCAL GRIDS: CONSTRUCTED FROM LOWER AND UPPER BOUNDS OF FULL SUB-DOMAIN
    xRange = blitz::Range(fullDomain.lbound(0), fullDomain.ubound(0));
    yRange = blitz::Range(fullDomain.lbound(1), fullDomain.ubound(1));
    zRange = blitz::Range(fullDomain.lbound(2), fullDomain.ubound(2));

    // LOCAL XI, ETA AND ZETA ARRAYS
    xi.resize(xRange);
    et.resize(yRange);
    zt.resize(zRange);

    // LOCAL GRID METRICS
    xi_x.resize(xRange);        xixx.resize(xRange);        xix2.resize(xRange);
    et_y.resize(yRange);        etyy.resize(yRange);        ety2.resize(yRange);
    zt_z.resize(zRange);        ztzz.resize(zRange);        ztz2.resize(zRange);

    // LOCAL GRID POINTS
#ifndef POST_RUN
    x.resize(xRange);
    y.resize(yRange);
    z.resize(zRange);
#else
    xRange = blitz::Range(coreDomain.lbound(0)-1, coreDomain.ubound(0)+1);
    yRange = blitz::Range(coreDomain.lbound(1)-1, coreDomain.ubound(1)+1);
    zRange = blitz::Range(coreDomain.lbound(2)-1, coreDomain.ubound(2)+1);

    x.resize(xRange);
    y.resize(yRange);
    z.resize(zRange);
#endif
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

    // ALONG XI-DIRECTION
    for (i = -padWidths(0); i < globalSize(0) + padWidths(0); i++) {
        xiGlo(i) = real(2*i + 1)*dXi/2;
    }

    // ALONG ETA-DIRECTION
    for (i = -padWidths(1); i < globalSize(1) + padWidths(1); i++) {
        etGlo(i) = real(2*i + 1)*dEt/2;
    }

    // ALONG ZETA-DIRECTION
    for (i = -padWidths(2); i < globalSize(2) + padWidths(2); i++) {
        ztGlo(i) = real(2*i + 1)*dZt/2;
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
    xixx = 0.0;
    xix2 = pow(xi_x, 2.0);

#ifndef PLANAR
    et_y = 1.0/yLen;
    etyy = 0.0;
    ety2 = pow(et_y, 2.0);
#endif

    zt_z = 1.0/zLen;
    ztzz = 0.0;
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
void grid::createTanHypGrid(int dim, blitz::Array<real, 1> xGlo, blitz::Array<real, 1> xiGl) {
    blitz::TinyVector<real, 3> dLen;
    blitz::Range lftPts, rgtPts, gmRange;
    blitz::Array<real, 1> df_x, dfxx, dfx2;
    std::vector<std::string> gTypeStr = {"single-sided", "double-sided"};
    auto gType = inputParams.xyzGrid[dim] - 1;

    dLen = xLen, yLen, zLen;

    // Hyperbolic tangent of beta parameter
    real thb = tanh(thBeta(dim));

    // Product of beta and length
    real btl = thBeta(dim)*dLen(dim);

    if (pf) {
        switch (dim) {
            case 0: std::cout << "Generating " << gTypeStr[gType] << " tangent hyperbolic grid along X direction" << std::endl;
                    break;
            case 1: std::cout << "Generating " << gTypeStr[gType] << " tangent hyperbolic grid along Y direction" << std::endl;
                    break;
            case 2: std::cout << "Generating " << gTypeStr[gType] << " tangent hyperbolic grid along Z direction" << std::endl;
                    break;
        }
    }

    // GENERATE X-GRID POINTS FROM UNIFORM XI-GRID POINTS AND THEIR METRICS
    df_x.resize(blitz::Range(-padWidths(dim), globalSize(dim) + padWidths(dim) - 1));
    dfxx.resize(blitz::Range(-padWidths(dim), globalSize(dim) + padWidths(dim) - 1));
    dfx2.resize(blitz::Range(-padWidths(dim), globalSize(dim) + padWidths(dim) - 1));

    for (int i = 0; i < globalSize(dim); i++) {
        if (gType) {
            // Double-sided tangent-hyperbolic stretching function
            xGlo(i) = dLen(dim)*(1.0 - tanh(thBeta[dim]*(1.0 - 2.0*xiGl(i)))/thb)/2.0;

            // Non-dimensionalized physical coordinate
            real ndx = xGlo(i)/dLen(dim);

            df_x(i) = thb/(btl*(1.0 - pow((1.0 - 2.0*ndx)*thb, 2)));
            dfxx(i) = -4.0*pow(thb, 3)*(1.0 - 2.0*ndx)/(dLen(dim)*btl*pow(1.0 - pow(thb*(1.0 - 2.0*ndx), 2), 2));
            dfx2(i) = pow(df_x(i), 2.0);
        } else {
            // Single-sided tangent-hyperbolic stretching function
            xGlo(i) = dLen(dim)*(1.0 - tanh(thBeta[dim]*(1.0 - xiGl(i)))/thb);

            // Non-dimensionalized physical coordinate
            real ndx = xGlo(i)/dLen(dim);

            df_x(i) = thb/(btl*(1.0 - pow((1.0 - ndx)*thb, 2)));
            dfxx(i) = -2.0*pow(thb, 3)*(1.0 - ndx)/(dLen(dim)*btl*pow(1.0 - pow(thb*(1.0 - ndx), 2), 2));
            dfx2(i) = pow(df_x(i), 2.0);
        }
    }

    lftPts = blitz::Range(-padWidths(dim), -1, 1);
    rgtPts = blitz::Range(globalSize(dim) - padWidths(dim), globalSize(dim) - 1, 1);

    xGlo(lftPts) = xGlo(rgtPts) - dLen(dim);
    df_x(lftPts) = df_x(rgtPts);
    dfxx(lftPts) = dfxx(rgtPts);
    dfx2(lftPts) = dfx2(rgtPts);

    rgtPts = blitz::Range(globalSize(dim), globalSize(dim) + padWidths(dim) - 1, 1);
    lftPts = blitz::Range(0, padWidths(dim) - 1, 1);

    xGlo(rgtPts) = dLen(dim) + xGlo(lftPts);
    df_x(rgtPts) = df_x(lftPts);
    dfxx(rgtPts) = dfxx(lftPts);
    dfx2(rgtPts) = dfx2(lftPts);

    gmRange = blitz::Range(globalMetrics(5*dim).lbound(0), globalMetrics(5*dim).ubound(0));
    globalMetrics(5*dim + 0) = xiGl(gmRange);
    globalMetrics(5*dim + 1) = xGlo(gmRange);
    globalMetrics(5*dim + 2) = df_x(gmRange);
    globalMetrics(5*dim + 3) = dfxx(gmRange);
    globalMetrics(5*dim + 4) = dfx2(gmRange);

    switch (dim) {
        case 0:
            xi_x = df_x(blitz::Range(subarrayStarts(dim) - padWidths(dim), subarrayEnds(dim) + padWidths(dim)));
            xixx = dfxx(blitz::Range(subarrayStarts(dim) - padWidths(dim), subarrayEnds(dim) + padWidths(dim)));
            xix2 = dfx2(blitz::Range(subarrayStarts(dim) - padWidths(dim), subarrayEnds(dim) + padWidths(dim)));
            break;
        case 1:
            et_y = df_x(blitz::Range(subarrayStarts(dim) - padWidths(dim), subarrayEnds(dim) + padWidths(dim)));
            etyy = dfxx(blitz::Range(subarrayStarts(dim) - padWidths(dim), subarrayEnds(dim) + padWidths(dim)));
            ety2 = dfx2(blitz::Range(subarrayStarts(dim) - padWidths(dim), subarrayEnds(dim) + padWidths(dim)));
            break;
        case 2:
            zt_z = df_x(blitz::Range(subarrayStarts(dim) - padWidths(dim), subarrayEnds(dim) + padWidths(dim)));
            ztzz = dfxx(blitz::Range(subarrayStarts(dim) - padWidths(dim), subarrayEnds(dim) + padWidths(dim)));
            ztz2 = dfx2(blitz::Range(subarrayStarts(dim) - padWidths(dim), subarrayEnds(dim) + padWidths(dim)));
            break;
    }

    mgGridMetrics(dim);
}


/**
 ********************************************************************************************************************************************
 * \brief   Function to initialize the globalMetrics array
 *
 *          The globalMetrics array contain the grid metric terms for the full domain along all 3 directions,
 *          for all the grids of the mult-grid V-Cycle.
 *          This array is used by the multi-grid Poisson solver to access grid-metric terms deep inside the
 *          neighbouring sub-domains at the coarsest levels of the V-Cycle.
 ********************************************************************************************************************************************
 */
void grid::mgGridMetrics() {
    blitz::TinyVector<real, 3> dLen;
    blitz::Range xRange, yRange, zRange;
    blitz::Array<real, 1> trnsGrid, physGrid;

    dLen = xLen, yLen, zLen;

    setVCDepth();

    // FOR EACH LEVEL, THERE ARE 15 ONE-DIMENSIONAL ARRAYS:
    // xi, x, xi_x, xixx, xix2,
    // et, y, et_y, etyy, ety2,
    // zt, z, zt_z, ztzz, ztz2
    globalMetrics.resize(15*(vcdGlo + 1));

    for (int vLev=0; vLev<=vcdGlo; vLev++) {
        int mgNx, mgNy, mgNz;

        mgNx = int(globalSize(0)/std::pow(2, vLev));
        mgNy = int(globalSize(1)/std::pow(2, vLev));
        mgNz = int(globalSize(2)/std::pow(2, vLev));

        // FIRST SET THE RANGES TO RESIZE ARRAYS
        xRange = blitz::Range(-1, mgNx);
        yRange = blitz::Range(-1, mgNy);
        zRange = blitz::Range(-1, mgNz);

        // START INDEX FOR LEVEL
        int ls = 15*vLev;

        // It has 9 arrays - xi_x, xixx, xix2, et_y, etyy, ety2, zt_z, ztzz, ztz2 in that order.
        for (int j=0;  j<5;  j++) globalMetrics(ls + j).resize(xRange);
        for (int j=5;  j<10; j++) globalMetrics(ls + j).resize(yRange);
        for (int j=10; j<15; j++) globalMetrics(ls + j).resize(zRange);

        if (vLev == 0) {
            // Uniform grid values at the finest grid level
            globalMetrics(0) = xiGlo(xRange);       globalMetrics(1) = xGlobal(xRange);
            globalMetrics(5) = etGlo(yRange);       globalMetrics(6) = yGlobal(yRange);
            globalMetrics(10) = ztGlo(zRange);      globalMetrics(11) = zGlobal(zRange);

        } else {
            for (int dim=0; dim<3; dim++) {
                int sls = ls + 5*dim;
                int cLen = globalMetrics(sls).ubound()(0);

                trnsGrid.resize(cLen + 2);      trnsGrid.reindexSelf(-1);
                physGrid.resize(cLen + 2);      physGrid.reindexSelf(-1);

                for (int i = 0; i < cLen; i++) {
                    trnsGrid(i) = (globalMetrics(sls - 15)(2*i) + globalMetrics(sls - 15)(2*i + 1))/2.0;
                    physGrid(i) = dLen(dim)*trnsGrid(i);
                }

                trnsGrid(-1) = -trnsGrid(0);
                physGrid(-1) = -physGrid(0);

                trnsGrid(cLen) = 1.0 + trnsGrid(0);
                physGrid(cLen) = dLen(dim) + physGrid(0);

                globalMetrics(sls) = trnsGrid;
                globalMetrics(sls + 1) = physGrid;
            }
        }

        // DEFAULT VALUES FOR UNIFORM GRID
        // THESE VALUES ARE OVERWRITTEN IF NON-UNIFORM GRID IS USED
        globalMetrics(ls +  2) = 1.0/xLen;     globalMetrics(ls +  3) = 0.0;     globalMetrics(ls +  4) = pow(globalMetrics(ls +  2), 2.0);
        globalMetrics(ls +  7) = 1.0/yLen;     globalMetrics(ls +  8) = 0.0;     globalMetrics(ls +  9) = pow(globalMetrics(ls +  7), 2.0);
        globalMetrics(ls + 12) = 1.0/zLen;     globalMetrics(ls + 13) = 0.0;     globalMetrics(ls + 14) = pow(globalMetrics(ls + 12), 2.0);
    }
}


/**
 ********************************************************************************************************************************************
 * \brief   Function to update the globalMetrics array for non-uniform grid
 *
 *          The overloaded version of this function is called to generate grid metrics for
 *          coarser grids of the multi-grid Poisson solver.
 ********************************************************************************************************************************************
 */
void grid::mgGridMetrics(int dim) {
    blitz::TinyVector<real, 3> dLen;
    blitz::Array<real, 1> trnsGrid, physGrid, df_x, dfxx, dfx2;
    auto gType = inputParams.xyzGrid[dim] - 1;

    dLen = xLen, yLen, zLen;

    // Hyperbolic tangent of beta parameter
    real thb = tanh(thBeta(dim));

    // Product of beta and length
    real btl = thBeta(dim)*dLen(dim);

    for (int vLev=1; vLev<=vcdGlo; vLev++) {
        // START INDEX FOR LEVEL
        int ls = 5*dim + 15*vLev;
        int cLen = globalMetrics(ls).shape()(0);

        // GENERATE X-GRID POINTS FROM UNIFORM XI-GRID POINTS AND THEIR METRICS
        physGrid.resize(cLen);      physGrid.reindexSelf(-1);
        df_x.resize(cLen);          df_x.reindexSelf(-1);
        dfxx.resize(cLen);          dfxx.reindexSelf(-1);
        dfx2.resize(cLen);          dfx2.reindexSelf(-1);

        for (int i = 0; i < cLen-2; i++) {
            if (gType) {
                // Double-sided tangent-hyperbolic stretching function
                physGrid(i) = dLen(dim)*(1.0 - tanh(thBeta(dim)*(1.0 - 2.0*globalMetrics(ls)(i)))/thb)/2.0;

                // Non-dimensionalized physical coordinate
                real ndx = physGrid(i)/dLen(dim);

                df_x(i) = thb/(btl*(1.0 - pow((1.0 - 2.0*ndx)*thb, 2)));
                dfxx(i) = -4.0*pow(thb, 3)*(1.0 - 2.0*ndx)/(dLen(dim)*btl*pow(1.0 - pow(thb*(1.0 - 2.0*ndx), 2), 2));
                dfx2(i) = pow(df_x(i), 2.0);
            } else {
                // Single-sided tangent-hyperbolic stretching function
                physGrid(i) = dLen(dim)*(1.0 - tanh(thBeta(dim)*(1.0 - globalMetrics(ls)(i)))/thb);

                // Non-dimensionalized physical coordinate
                real ndx = physGrid(i)/dLen(dim);

                df_x(i) = thb/(btl*(1.0 - pow((1.0 - ndx)*thb, 2)));
                dfxx(i) = -2.0*pow(thb, 3)*(1.0 - ndx)/(dLen(dim)*btl*pow(1.0 - pow(thb*(1.0 - ndx), 2), 2));
                dfx2(i) = pow(df_x(i), 2.0);
            }
        }

        physGrid(-1) = -physGrid(0);
        df_x(-1) = df_x(cLen-3);
        dfxx(-1) = dfxx(cLen-3);
        dfx2(-1) = dfx2(cLen-3);

        physGrid(cLen-2) = dLen(dim) + physGrid(0);
        df_x(cLen-2) = df_x(0);
        dfxx(cLen-2) = dfxx(0);
        dfx2(cLen-2) = dfx2(0);

        globalMetrics(ls + 1) = physGrid;
        globalMetrics(ls + 2) = df_x;
        globalMetrics(ls + 3) = dfxx;
        globalMetrics(ls + 4) = dfx2;
    }
}


/**
 ********************************************************************************************************************************************
 * \brief   Function to set local and global V-cycle depths
 *
 *          The function computes the maximum allowable V-cycle depth for the given grid configuration.
 *          There are two allowable depths at the local and global levels for a given MPI decomposition.
 *          Both values are calculated here.
 ********************************************************************************************************************************************
 */
void grid::setVCDepth() {
    int divNum;

#ifdef PLANAR
    blitz::TinyVector<int, 2> maxSize, remList, subSize;
#else
    blitz::TinyVector<int, 3> maxSize, remList, subSize;
#endif

    /////////////// First set global depth ///////////////
    vcdGlo = 0;
#ifdef PLANAR
    maxSize = globalSize(0), globalSize(2);
#else
    maxSize = globalSize;
#endif
    while (true) {
        divNum = int(std::pow(2, vcdGlo));
        remList = maxSize % divNum;
        subSize = maxSize / divNum;
        if (blitz::max(remList) > 0) break;
        vcdGlo += 1;
    }
    vcdGlo -= 1;

    // Check if there is only one point along any direction in sub-domains.
    // If so, then decrease the V-Cycle depth by 1 to prevent it.
    divNum = int(std::pow(2, vcdGlo));
    subSize = maxSize / divNum;
    if (blitz::min(subSize) == 1) vcdGlo -= 1;

    // This additional adjustment is necessary when domain is non-periodic along all directions.
    // Due to the resultant Neumann BC in pressure correction, multi-drid doesn't
    // perform well at the coarsest level. Hence decrease the depth by 1 for this case.
#ifdef PLANAR
    if ((not inputParams.xPer) and (not inputParams.zPer)) vcdGlo -= 1;
#else
    if ((not inputParams.xPer) and (not inputParams.yPer) and (not inputParams.zPer)) vcdGlo -= 1;
#endif

    /////////////// Then set local depth ///////////////
    vcdLoc = 0;
#ifdef PLANAR
    maxSize = coreSize(0), coreSize(2);
#else
    maxSize = coreSize;
#endif
    while (true) {
        divNum = int(std::pow(2, vcdLoc));
        remList = maxSize % divNum;
        subSize = maxSize / divNum;
        if (blitz::max(remList) > 0) break;
        vcdLoc += 1;
    }
    vcdLoc -= 1;

    // Check if there is only one point along any direction in sub-domains.
    // If so, then decrease the V-Cycle depth by 1 to prevent it.
    divNum = int(std::pow(2, vcdLoc));
    subSize = maxSize / divNum;
    if (blitz::min(subSize) == 1) vcdLoc -= 1;

    // Final check for non-periodic serial runs
    if (vcdLoc > vcdGlo) vcdLoc = vcdGlo;
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
        if (pf) std::cout << "\nWARNING: Grid anisotropy exceeds limits. Finite-difference calculations will be inaccurate" << std::endl;
    } else {
        if (pf) std::cout << "\nMaximum grid anisotropy is " << globalMax << std::endl;
    }
}

