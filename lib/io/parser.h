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
/*! \file parser.h
 *
 *  \brief Class declaration of parser
 *
 *  \author Roshan Samuel, Shashwat Bhattacharya
 *  \date Nov 2019
 *  \copyright New BSD License
 *
 ********************************************************************************************************************************************
 */

#ifndef PARSER_H
#define PARSER_H

#include <math.h>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <blitz/array.h>
#include <yaml-cpp/yaml.h>

#ifdef REAL_DOUBLE
#define H5T_NATIVE_REAL H5T_NATIVE_DOUBLE
#define MPI_FP_REAL MPI_DOUBLE
#define real double
#else
#define H5T_NATIVE_REAL H5T_NATIVE_FLOAT
#define MPI_FP_REAL MPI_FLOAT
#define real float
#endif

class parser {
    public:
        int ioCnt;
        int rbcType;
        int nThreads;
        int forceType;
        int solnFormat;
        int Nx, Ny, Nz;
        int npX, npY, npZ;
        int resType, vcCount;
        int preSmooth, postSmooth;

        int icType;
        int dScheme;
        int iScheme;
        int nlScheme;
        int lesModel;
        int probType;

        bool useCFL;
        bool nonHgBC;
        bool solveFlag;
        bool readProbes;
        bool useCustomBC;
        bool restartFlag;
        bool printResidual;
        bool xPer, yPer, zPer;

        real Re;
        real Ra;
        real Pr;
        real Ta;
        real Ro;

        real fwInt;
        real rsInt;
        real prInt;
        real sorParam;
        real meanPGrad;
        real Lx, Ly, Lz;
        real tStp, tMax;
        real patchRadius;
        real rfIntensity;
        real meanVelocity;
        real courantNumber;
        real peLimit, upParam;
        real betaX, betaY, betaZ;
        real cnTolerance, mgTolerance, vcTolerance;

        std::vector<int> xyzGrid;
        blitz::TinyVector<real, 3> gAxis, rAxis;
        std::vector<blitz::TinyVector<int, 3> > probesList;

        parser();

        void writeParams();
        std::vector<int> getBCTypes(std::string vStr, std::string fStr) const;
        std::vector<real> getBCValues(std::string vStr, std::string fStr) const;

        std::vector<real> readTimes();

    private:
        std::string meshType;
        std::string domainType;
        std::string probeCoords;

#ifdef PLANAR
        const std::vector<std::string> sList = {
            "Left Wall",
            "Right Wall",
            "Bottom Wall",
            "Top Wall"
        };
#else
        const std::vector<std::string> sList = {
            "Left Wall",
            "Right Wall",
            "Front Wall",
            "Back Wall",
            "Bottom Wall",
            "Top Wall"
        };
#endif

        void parseYAML();
        void checkData();

        void testProbes();
        void parseProbes();

        void setGrids();
        void setPeriodicity();
};

/**
 ********************************************************************************************************************************************
 *  \class parser parser.h "lib/io/parser.h"
 *  \brief  Contains all the global variables set by the user through the yaml file
 *
 *  The class parses the paramters.yaml file and stores all the simulation paramters in publicly accessible constants.
 *  The class also has a function to check the consistency of the user set paramters and throw exceptions.
 *  The class is best initialized as a constant to prevent inadvertent tampering of the global variables it contains.
 ********************************************************************************************************************************************
 */

#endif
