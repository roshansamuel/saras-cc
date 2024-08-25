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
/*! \file parser.cc
 *
 *  \brief Definitions for functions of class parser
 *  \sa parser.h
 *  \author Roshan Samuel, Shashwat Bhattacharya
 *  \date Nov 2019
 *  \copyright New BSD License
 *
 ********************************************************************************************************************************************
 */

#include <iostream>
#include "parser.h"
#include "mpi.h"

parser::parser() {
    parseYAML();
    checkData();

    setGrids();
    setPeriodicity();

    if (readProbes) {
        parseProbes();

        testProbes();
    }
}

/**
 ********************************************************************************************************************************************
 * \brief   Function to open the yaml file and parse the parameters
 *
 *          The function opens the parameters.yaml file and parses the simulation parameters into its member variables that are publicly
 *          accessible.
 ********************************************************************************************************************************************
 */
void parser::parseYAML() {
    std::ifstream inFile;

    inFile.open("input/parameters.yaml", std::ifstream::in);

#ifdef YAML_LEGACY
    real gx, gy, gz;
    real rx, ry, rz;

    YAML::Node yamlNode;
    YAML::Parser parser(inFile);

    parser.GetNextDocument(yamlNode);

    /********** Problem parameters **********/

    yamlNode["Program"]["Problem Type"] >> probType;
    yamlNode["Program"]["Initial Condition"] >> icType;
    yamlNode["Program"]["Mean Flow Velocity"] >> meanVelocity;
    yamlNode["Program"]["Perturbation Intensity"] >> rfIntensity;
    yamlNode["Program"]["Domain Type"] >> domainType;
    yamlNode["Program"]["Custom BCs"] >> useCustomBC;
    yamlNode["Program"]["RBC Type"] >> rbcType;

    yamlNode["Program"]["LES Model"] >> lesModel;

    yamlNode["Program"]["Reynolds Number"] >> Re;
    yamlNode["Program"]["Rossby Number"] >> Ro;
    yamlNode["Program"]["Rayleigh Number"] >> Ra;
    yamlNode["Program"]["Prandtl Number"] >> Pr;
    yamlNode["Program"]["Taylor Number"] >> Ta;

    yamlNode["Program"]["X Length"] >> Lx;
    yamlNode["Program"]["Y Length"] >> Ly;
    yamlNode["Program"]["Z Length"] >> Lz;

    yamlNode["Program"]["Heating Plate"] >> nonHgBC;
    yamlNode["Program"]["Plate Radius"] >> patchRadius;

    yamlNode["Program"]["Force"] >> forceType;

    yamlNode["Program"]["Gravity Direction"][0] >> gx;
    yamlNode["Program"]["Gravity Direction"][1] >> gy;
    yamlNode["Program"]["Gravity Direction"][2] >> gz;

    gAxis = gx, gy, gz;

    yamlNode["Program"]["Rotation Axis"][0] >> rx;
    yamlNode["Program"]["Rotation Axis"][1] >> ry;
    yamlNode["Program"]["Rotation Axis"][2] >> rz;

    rAxis = rx, ry, rz;

    yamlNode["Program"]["Mean Pressure Gradient"] >> meanPGrad;

    /********** Mesh parameters **********/

    yamlNode["Mesh"]["Mesh Type"] >> meshType;

    yamlNode["Mesh"]["X Beta"] >> betaX;
    yamlNode["Mesh"]["Y Beta"] >> betaY;
    yamlNode["Mesh"]["Z Beta"] >> betaZ;

    yamlNode["Mesh"]["X Size"] >> Nx;
    yamlNode["Mesh"]["Y Size"] >> Ny;
    yamlNode["Mesh"]["Z Size"] >> Nz;

    /********** Parallelization parameters **********/

    yamlNode["Parallel"]["Number of OMP threads"] >> nThreads;

    yamlNode["Parallel"]["X Number of Procs"] >> npX;
    yamlNode["Parallel"]["Y Number of Procs"] >> npY;
    yamlNode["Parallel"]["Z Number of Procs"] >> npZ;

    /********** Solver parameters **********/

    yamlNode["Solver"]["Differentiation Scheme"] >> dScheme;
    yamlNode["Solver"]["Non-Linear Term"] >> nlScheme;
    yamlNode["Solver"]["Peclet Limit"] >> peLimit;
    yamlNode["Solver"]["Central Bias"] >> upParam;

    yamlNode["Solver"]["Integration Scheme"] >> iScheme;
    yamlNode["Solver"]["Solve Tolerance"] >> cnTolerance;

    yamlNode["Solver"]["Restart Run"] >> restartFlag;

    yamlNode["Solver"]["Use CFL Condition"] >> useCFL;
    yamlNode["Solver"]["Courant Number"] >> courantNumber;
    yamlNode["Solver"]["Time-Step"] >> tStp;
    yamlNode["Solver"]["Final Time"] >> tMax;

    yamlNode["Solver"]["I/O Count"] >> ioCnt;
    yamlNode["Solver"]["Solution Format"] >> solnFormat;
    yamlNode["Solver"]["Solution Write Interval"] >> fwInt;
    yamlNode["Solver"]["Restart Write Interval"] >> rsInt;

    yamlNode["Solver"]["Record Probes"] >> readProbes;
    yamlNode["Solver"]["Probe Time Interval"] >> prInt;
    yamlNode["Solver"]["Probes"] >> probeCoords;

    /********** Multigrid parameters **********/

    yamlNode["Multigrid"]["V-Cycle Count"] >> vcCount;
    yamlNode["Multigrid"]["Residual Tolerance"] >> vcTolerance;

    yamlNode["Multigrid"]["Solve Coarsest"] >> solveFlag;
    yamlNode["Multigrid"]["Solve Tolerance"] >> mgTolerance;
    yamlNode["Multigrid"]["SOR Parameter"] >> sorParam;

    yamlNode["Multigrid"]["Pre-Smoothing Count"] >> preSmooth;
    yamlNode["Multigrid"]["Post-Smoothing Count"] >> postSmooth;

    yamlNode["Multigrid"]["Residual Type"] >> resType;
    yamlNode["Multigrid"]["Print Residual"] >> printResidual;

#else
    YAML::Node yamlNode = YAML::Load(inFile);

    /********** Problem parameters **********/

    probType = yamlNode["Program"]["Problem Type"].as<int>();
    icType = yamlNode["Program"]["Initial Condition"].as<int>();
    meanVelocity = yamlNode["Program"]["Mean Flow Velocity"].as<real>();
    rfIntensity = yamlNode["Program"]["Perturbation Intensity"].as<real>();
    domainType = yamlNode["Program"]["Domain Type"].as<std::string>();
    useCustomBC = yamlNode["Program"]["Custom BCs"].as<bool>();
    rbcType = yamlNode["Program"]["RBC Type"].as<int>();

    lesModel = yamlNode["Program"]["LES Model"].as<int>();

    Re = yamlNode["Program"]["Reynolds Number"].as<real>();
    Ro = yamlNode["Program"]["Rossby Number"].as<real>();
    Ra = yamlNode["Program"]["Rayleigh Number"].as<real>();
    Pr = yamlNode["Program"]["Prandtl Number"].as<real>();
    Ta = yamlNode["Program"]["Taylor Number"].as<real>();

    Lx = yamlNode["Program"]["X Length"].as<real>();
    Ly = yamlNode["Program"]["Y Length"].as<real>();
    Lz = yamlNode["Program"]["Z Length"].as<real>();

    nonHgBC = yamlNode["Program"]["Heating Plate"].as<bool>();
    patchRadius = yamlNode["Program"]["Plate Radius"].as<real>();

    forceType = yamlNode["Program"]["Force"].as<int>();

    gAxis = yamlNode["Program"]["Gravity Direction"][0].as<real>(),
            yamlNode["Program"]["Gravity Direction"][1].as<real>(),
            yamlNode["Program"]["Gravity Direction"][2].as<real>();

    rAxis = yamlNode["Program"]["Rotation Axis"][0].as<real>(),
            yamlNode["Program"]["Rotation Axis"][1].as<real>(),
            yamlNode["Program"]["Rotation Axis"][2].as<real>();

    meanPGrad = yamlNode["Program"]["Mean Pressure Gradient"].as<real>();

    /********** Mesh parameters **********/

    meshType = yamlNode["Mesh"]["Mesh Type"].as<std::string>();

    betaX = yamlNode["Mesh"]["X Beta"].as<real>();
    betaY = yamlNode["Mesh"]["Y Beta"].as<real>();
    betaZ = yamlNode["Mesh"]["Z Beta"].as<real>();

    Nx = yamlNode["Mesh"]["X Size"].as<int>();
    Ny = yamlNode["Mesh"]["Y Size"].as<int>();
    Nz = yamlNode["Mesh"]["Z Size"].as<int>();

    /********** Parallelization parameters **********/

    nThreads = yamlNode["Parallel"]["Number of OMP threads"].as<int>();

    npX = yamlNode["Parallel"]["X Number of Procs"].as<int>();
    npY = yamlNode["Parallel"]["Y Number of Procs"].as<int>();
    npZ = yamlNode["Parallel"]["Z Number of Procs"].as<int>();

    /********** Solver parameters **********/

    dScheme = yamlNode["Solver"]["Differentiation Scheme"].as<int>();
    nlScheme = yamlNode["Solver"]["Non-Linear Term"].as<int>();
    peLimit = yamlNode["Solver"]["Peclet Limit"].as<real>();
    upParam = yamlNode["Solver"]["Central Bias"].as<real>();

    iScheme = yamlNode["Solver"]["Integration Scheme"].as<int>();
    cnTolerance = yamlNode["Solver"]["Solve Tolerance"].as<real>();

    restartFlag = yamlNode["Solver"]["Restart Run"].as<bool>();

    useCFL = yamlNode["Solver"]["Use CFL Condition"].as<bool>();
    courantNumber = yamlNode["Solver"]["Courant Number"].as<real>();
    tStp = yamlNode["Solver"]["Time-Step"].as<real>();
    tMax = yamlNode["Solver"]["Final Time"].as<real>();

    ioCnt = yamlNode["Solver"]["I/O Count"].as<int>();
    solnFormat = yamlNode["Solver"]["Solution Format"].as<int>();
    fwInt = yamlNode["Solver"]["Solution Write Interval"].as<real>();
    rsInt = yamlNode["Solver"]["Restart Write Interval"].as<real>();

    readProbes = yamlNode["Solver"]["Record Probes"].as<bool>();
    prInt = yamlNode["Solver"]["Probe Time Interval"].as<real>();
    probeCoords = yamlNode["Solver"]["Probes"].as<std::string>();

    /********** Multigrid parameters **********/

    vcCount = yamlNode["Multigrid"]["V-Cycle Count"].as<int>();
    vcTolerance = yamlNode["Multigrid"]["Residual Tolerance"].as<real>();

    solveFlag = yamlNode["Multigrid"]["Solve Coarsest"].as<bool>();
    mgTolerance = yamlNode["Multigrid"]["Solve Tolerance"].as<real>();
    sorParam = yamlNode["Multigrid"]["SOR Parameter"].as<real>();

    preSmooth = yamlNode["Multigrid"]["Pre-Smoothing Count"].as<int>();
    postSmooth = yamlNode["Multigrid"]["Post-Smoothing Count"].as<int>();

    resType = yamlNode["Multigrid"]["Residual Type"].as<int>();
    printResidual = yamlNode["Multigrid"]["Print Residual"].as<bool>();
#endif

    inFile.close();
}

/**
 ********************************************************************************************************************************************
 * \brief   Function to perform a check on the consistency of user-set parameters
 *
 *          In order to catch potential errors early on, a few basic checks are performed here to validate the paramters set
 *          by the user.
 *          Additional checks to be performed on the paramters can be added to this function if necessary.
 ********************************************************************************************************************************************
 */
void parser::checkData() {
    // CHECK IF GRID SIZES ARE SET CORRECTLY FOR A 2D/3D SIMULATION
    if (Nx < 2) {
        std::cout << "ERROR: Insufficient number of points along X axis. Aborting" << std::endl;
        MPI_Finalize();
        exit(0);
    }
#ifdef PLANAR
    if (Ny != 1) {
        std::cout << "WARNING: The number of points along Y is greater than 1 although solver has been compiled with PLANAR flag. Resetting Y Size to 1" << std::endl;
        Ny = 1;
    }
#else
    if (Ny == 1) {
        std::cout << "ERROR: Insufficient number of points along Y axis. Aborting" << std::endl;
        MPI_Finalize();
        exit(0);
    }
#endif
    if (Nz < 2) {
        std::cout << "ERROR: Insufficient number of points along Z axis. Aborting" << std::endl;
        MPI_Finalize();
        exit(0);
    }

    // CHECK IF LESS THAN 1 PROCESSOR IS ASKED FOR ALONG X-DIRECTION. IF SO, WARN AND SET IT TO DEFAULT VALUE OF 1
    if (npX < 1) {
        std::cout << "WARNING: Number of processors in X-direction is less than 1. Setting it to 1" << std::endl;
        npX = 1;
    }

    // CHECK IF LESS THAN 1 PROCESSOR IS ASKED FOR ALONG Y-DIRECTION. IF SO, WARN AND SET IT TO DEFAULT VALUE OF 1
    if (npY < 1) {
        std::cout << "WARNING: Number of processors in Y-direction is less than 1. Setting it to 1" << std::endl;
        npY = 1;
    }

    // CHECK IF LESS THAN 1 PROCESSOR IS ASKED FOR ALONG Z-DIRECTION. IF SO, WARN AND SET IT TO DEFAULT VALUE OF 1
    if (npZ < 1) {
        std::cout << "WARNING: Number of processors in Z-direction is less than 1. Setting it to 1" << std::endl;
        npZ = 1;
    }

    // CHECK IF DOMAIN TYPE STRING IS OF CORRECT LENGTH
    if (domainType.length() != 3) {
        std::cout << "ERROR: Domain type string is not correct. Aborting" << std::endl;
        MPI_Finalize();
        exit(0);
    }

    // CHECK IF MESH TYPE STRING IS OF CORRECT LENGTH
    if (meshType.length() != 3) {
        std::cout << "ERROR: Mesh type string is not correct. Aborting" << std::endl;
        MPI_Finalize();
        exit(0);
    }

    // CHECK IF THE I/O COUNT IS VALID
    if (ioCnt < 1) {
        std::cout << "WARNING: I/O Count parameter must be a positive integer. Setting it default value of 1" << std::endl;
        ioCnt = 1;
    }

    // CHECK IF THE TIME-STEP SET BY USER IS LESS THAN THE MAXIMUM TIME SPECIFIED FOR SIMULATION.
    if (tStp > tMax) {
        std::cout << "ERROR: Time step is larger than the maximum duration assigned for simulation. Aborting" << std::endl;
        MPI_Finalize();
        exit(0);
    }

    // CHECK IF MORE THAN 1 PROCESSOR IS ASKED FOR ALONG Y-DIRECTION FOR A 2D SIMULATION
#ifdef PLANAR
    if (npY > 1) {
        std::cout << "WARNING: More than 1 processor is specified along Y-direction although the PLANAR flag is set. Setting npY to 1" << std::endl;
        npY = 1;
    }
#endif

#ifdef REAL_SINGLE
    if ((cnTolerance < 5.0e-6) or (mgTolerance < 5.0e-6)) {
        std::cout << "ERROR: The specified tolerance for iterative solvers is too small for single precision calculations. Aborting" << std::endl;
        MPI_Finalize();
        exit(0);
    }
#endif

    if (resType > 2) {
        std::cout << "ERROR: The specified value for printing error at end of V-Cycles is not defined. Aborting" << std::endl;
        MPI_Finalize();
        exit(0);
    }

    if ((probType < 5) and (lesModel == 2)) {
        std::cout << "WARNING: The specified LES Model is incompatible with the problem type. Resetting LES Model to 1" << std::endl;
        lesModel = 1;
    }

    // CHECK IF THE SOLUTION FORMAT SPECIFIED BY THE USER IS NOT CONSISTENT
    if (solnFormat > 2) {
        std::cout << "ERROR: The specified Solution Format is inconsistent with expected values. Aborting" << std::endl;
        MPI_Finalize();
        exit(0);
    }
}

/**
 ********************************************************************************************************************************************
 * \brief   Function to set the grid types along each direction based on meshType variable
 *
 *          The user specifies mesh type as a single string.
 *          This string has to be parsed to set the integer values xGrid, yGrid and zGrid.
 *          The values of these variables will determine the grid stretching along each direction appropriately.
 ********************************************************************************************************************************************
 */
void parser::setGrids() {
    int xGrid, yGrid, zGrid;

    // The integer values xGrid, yGrid and zGrid are set as below:
    // 0 - uniform grid
    // 1 - single-sided tangent-hyperbolic stretching
    // 2 - double-sided tangent-hyperbolic stretching
    xGrid = 0;
    yGrid = 0;
    zGrid = 0;

    char charMTypes[4];
    std::strcpy(charMTypes, meshType.c_str());

    switch (charMTypes[0]) {
        case 'S': xGrid = 1;
            break;
        case 'D': xGrid = 2;
            break;
    }

    switch (charMTypes[1]) {
        case 'S': yGrid = 1;
            break;
        case 'D': yGrid = 2;
            break;
    }

    switch (charMTypes[2]) {
        case 'S': zGrid = 1;
            break;
        case 'D': zGrid = 2;
            break;
    }

    xyzGrid = {xGrid, yGrid, zGrid};
}

/**
 ********************************************************************************************************************************************
 * \brief   Function to set the periodicity of the domain based on domainType variable
 *
 *          The user specifies domain type as a single string.
 *          This string has to be parsed to set the boolean values xPer, yPer and zPer.
 *          The values of these variables will set the periodic boundary conditions appropriately.
 ********************************************************************************************************************************************
 */
void parser::setPeriodicity() {
    xPer = true;
    yPer = true;
    zPer = true;

    if (domainType[0] == 'N') xPer = false;
    if (domainType[1] == 'N') yPer = false;
    if (domainType[2] == 'N') zPer = false;
}

/**
 ********************************************************************************************************************************************
 * \brief   Function to parse the probeCoords string
 *
 *          The user specifies probeCoords in NumPy's linspace style with colons
 *          This function extracts the coordinates of the probes from the given string.
 ********************************************************************************************************************************************
 */
void parser::parseProbes() {
    std::string errorProbe;

    while (true) {
        std::vector<std::vector<int> > indexList;

        // Extract the leading set enclosed by square brackets
        std::string oneSet = probeCoords.substr(probeCoords.find('[') + 1, probeCoords.find(']') - 1);
        errorProbe = oneSet;

        oneSet.append(",");
        while (true) {
            std::vector<int> indexVector;
            std::istringstream iss;

            std::string indexData = oneSet.substr(oneSet.find_first_not_of(' '), oneSet.find(',') - oneSet.find_first_not_of(' '));
            indexData.erase(indexData.find_last_not_of(' ') + 1, indexData.length());

            // Erase the extracted set
            oneSet.erase(0, oneSet.find(',') + 1);

            if (indexData.find(":") == std::string::npos) {
                int indexVal;

                // Only a single integer
                iss.str(indexData);
                iss >> indexVal;
                indexVector.push_back(indexVal);
                indexList.push_back(indexVector);

            } else {
                unsigned int strIndex, endIndex, numIndex;

                // Further processing necessary to extract range
                std::string rangeStart = indexData.substr(0, indexData.find(':'));
                indexData.erase(0, indexData.find(':') + 1);
                iss.clear();
                iss.str(rangeStart);
                iss >> strIndex;

                std::string rangeEnd = indexData.substr(0, indexData.find(':'));
                indexData.erase(0, indexData.find(':') + 1);
                iss.clear();
                iss.str(rangeEnd);
                iss >> endIndex;

                iss.clear();
                iss.str(indexData);
                iss >> numIndex;

                if ((strIndex == endIndex) or (numIndex == 1)) {
                    indexVector.push_back(strIndex);

                } else {
                    for (unsigned int i = 0; i < numIndex; i++) {
                        real incIndex = ((real)endIndex - (real)strIndex)/((real)numIndex - 1);
                        int probeIndex = strIndex + (int)round((real)i*incIndex);
                        indexVector.push_back(probeIndex);
                    }
                }

                indexList.push_back(indexVector);
            }

            if (oneSet.length() < 2) {
                break;
            }
        }

        // Erase the extracted set
        probeCoords.erase(0, probeCoords.find(']') + 1);

        // Remove possible white-spaces between the sets
        probeCoords.erase(0, probeCoords.find_first_not_of(' '));

        // Add the coordinates from the extracted set to global coordinates list
#ifdef PLANAR
        if (indexList.size() == 2) {
            for (unsigned int iX = 0; iX < indexList[0].size(); iX++) {
                for (unsigned int iZ = 0; iZ < indexList[1].size(); iZ++) {
                    blitz::TinyVector<int, 3> probeLoc;
                    probeLoc = indexList[0][iX], 0, indexList[1][iZ];
                    probesList.push_back(probeLoc);
                }
            }
        } else if (indexList.size() == 3) {
            for (unsigned int iX = 0; iX < indexList[0].size(); iX++) {
                for (unsigned int iZ = 0; iZ < indexList[2].size(); iZ++) {
                    blitz::TinyVector<int, 3> probeLoc;
                    probeLoc = indexList[0][iX], 0, indexList[2][iZ];
                    probesList.push_back(probeLoc);
                }
            }
        } else {
            std::cout << "WARNING: Number of indices for the probe(s) " << errorProbe << " does not match dimensionality of problem." << std::endl;
        }
#else
        if (indexList.size() == 3) {
            for (unsigned int iX = 0; iX < indexList[0].size(); iX++) {
                for (unsigned int iY = 0; iY < indexList[1].size(); iY++) {
                    for (unsigned int iZ = 0; iZ < indexList[2].size(); iZ++) {
                        blitz::TinyVector<int, 3> probeLoc;
                        probeLoc = indexList[0][iX], indexList[1][iY], indexList[2][iZ];
                        probesList.push_back(probeLoc);
                    }
                }
            }
        } else {
            std::cout << "ERROR: Number of indices for the probe(s) [" << errorProbe << "] does not match dimensionality of problem. Aborting" << std::endl;
            MPI_Finalize();
            exit(0);
        }
#endif

        // The number 3 is randomly chosen. Ideally if the string is smaller than that, it has no more info
        if (probeCoords.length() < 3) {
            break;
        }
    }
}

/**
 ********************************************************************************************************************************************
 * \brief   Function to test if the probes specified by user are valid
 *
 *          All the probe indices should lie within the domain limits.
 *          This function performs this check to avoid unpleasant surprises later on.
 ********************************************************************************************************************************************
 */
void parser::testProbes() {
    for (unsigned int i = 0; i < probesList.size(); i++) {
        if (probesList[i][0] < 0 or probesList[i][0] > Nx) {
            std::cout << "ERROR: The X index of the probe " << probesList[i] << " lies outside the bounds of the domain. Aborting" << std::endl;
            MPI_Finalize();
            exit(0);
        }

#ifndef PLANAR
        if (probesList[i][1] < 0 or probesList[i][1] > Ny) {
            std::cout << "ERROR: The Y index of the probe " << probesList[i] << " lies outside the bounds of the domain. Aborting" << std::endl;
            MPI_Finalize();
            exit(0);
        }
#endif

        if (probesList[i][2] < 0 or probesList[i][2] > Nz) {
            std::cout << "ERROR: The Z index of the probe " << probesList[i] << " lies outside the bounds of the domain. Aborting" << std::endl;
            MPI_Finalize();
            exit(0);
        }
    }
}

/**
 ********************************************************************************************************************************************
 * \brief   Function to write all the parameter values to I/O
 *
 *          All the user set parameters have to be written to I/O so that the runlog or any out file
 *          contains all the relevant information about the case that was run.
 *          This public function has to be called from the solver by one rank only.
 ********************************************************************************************************************************************
 */
void parser::writeParams() {
    std::cout << std::endl << "Writing all parameters from the YAML input file for reference" << std::endl << std::endl;
    std::cout << "\t****************** START OF parameters.yaml ******************" << std::endl << std::endl;

    std::ifstream inFile;
    inFile.open("input/parameters.yaml", std::ifstream::in);
    std::string line;
    while (std::getline(inFile, line)) {
        std::cout << line << std::endl;
    }
    inFile.close();

    std::cout << std::endl << "\t******************* END OF parameters.yaml *******************" << std::endl;
    std::cout << std::endl;

    if (useCustomBC) {
        std::cout << std::endl << "Use of custom BCs is enabled" << std::endl;
        std::cout << std::endl << "Writing BC parameters from the YAML file for reference" << std::endl << std::endl;
        std::cout << "\t****************** START OF customBCs.yaml ******************" << std::endl << std::endl;

        std::ifstream inFile;
        inFile.open("input/customBCs.yaml", std::ifstream::in);
        std::string line;
        while (std::getline(inFile, line)) {
            std::cout << line << std::endl;
        }
        inFile.close();

        std::cout << std::endl << "\t******************* END OF customBCs.yaml *******************" << std::endl;
        std::cout << std::endl;
    }
}

/**
 ********************************************************************************************************************************************
 * \brief   Function to read a list of solution times for which solution files are available
 *
 *          This function is called only for post-processing runs, where a list of solution times
 *          is given, such that a solution file exists for each entry in the list.
 *          This list is read and returned by the function.
 ********************************************************************************************************************************************
 */
std::vector<real> parser::readTimes() {
    std::vector<real> timeList;
    std::ifstream tlFile;
    std::string tString;
    real tVal;

    tlFile.open("output/timeList.dat");
    if (tlFile.is_open()) {
        while (std::getline(tlFile, tString)) {
            tVal = atof(tString.c_str());
            timeList.push_back(tVal);
        }
    } else {
        std::cout << "ERROR: Could not read list of solution times. Aborting" << std::endl;
        MPI_Finalize();
        exit(0);
    }

    return timeList;
}

/**
 ********************************************************************************************************************************************
 * \brief   Function to parse the customBCs YAML file and obtain the list of BC types
 *
 *          This function parses the customBCs.yaml file and returns the list of
 *          BC types (neumann/dirichlet) for a given variable name and field name.
 ********************************************************************************************************************************************
 */
std::vector<int> parser::getBCTypes(std::string vStr, std::string fStr) const {
    std::vector<int> bcTypes;
    std::ifstream inFile;
    int inpType;

    inFile.open("input/customBCs.yaml", std::ifstream::in);
    if (not inFile.is_open()) {
        std::cout << "ERROR: Could not read customBCs.yaml. Aborting" << std::endl;
        MPI_Finalize();
        exit(0);
    }

#ifdef YAML_LEGACY
    YAML::Node yamlNode;
    YAML::Parser parser(inFile);

    parser.GetNextDocument(yamlNode);

    for (const std::string& sStr : sList) {
        yamlNode[vStr][fStr][sStr]["BC Type"] >> inpType;
        bcTypes.push_back(inpType);
    }
#else
    YAML::Node yamlNode = YAML::Load(inFile);

    for (const std::string& sStr : sList) {
        inpType = yamlNode[vStr][fStr][sStr]["BC Type"].as<int>();
        bcTypes.push_back(inpType);
    }
#endif
    inFile.close();

    return bcTypes;
}

/**
 ********************************************************************************************************************************************
 * \brief   Function to parse the customBCs YAML file and obtain the list of BC values
 *
 *          This function parses the customBCs.yaml file and returns the list of
 *          BC values corresponding to the BC types for a given variable name and field name.
 ********************************************************************************************************************************************
 */
std::vector<real> parser::getBCValues(std::string vStr, std::string fStr) const {
    std::vector<real> bcValues;
    std::ifstream inFile;
    real inpVal;

    inFile.open("input/customBCs.yaml", std::ifstream::in);
    if (not inFile.is_open()) {
        std::cout << "ERROR: Could not read customBCs.yaml. Aborting" << std::endl;
        MPI_Finalize();
        exit(0);
    }

#ifdef YAML_LEGACY
    YAML::Node yamlNode;
    YAML::Parser parser(inFile);

    parser.GetNextDocument(yamlNode);

    for (const std::string& sStr : sList) {
        yamlNode[vStr][fStr][sStr]["BC Value"] >> inpVal;
        bcValues.push_back(inpVal);
    }
#else
    YAML::Node yamlNode = YAML::Load(inFile);

    for (const std::string& sStr : sList) {
        inpVal = yamlNode[vStr][fStr][sStr]["BC Value"].as<real>();
        bcValues.push_back(inpVal);
    }
#endif
    inFile.close();

    return bcValues;
}
