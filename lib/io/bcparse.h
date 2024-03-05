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
/*! \file bcparse.h
 *
 *  \brief Class declaration of boundary condition parser
 *
 *  \author Roshan Samuel
 *  \date Mar 2024
 *  \copyright New BSD License
 *
 ********************************************************************************************************************************************
 */

#ifndef BCPARSE_H
#define BCPARSE_H

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

class bcparse {
    public:
        std::vector<int> uBCType, vBCType, wBCType, tBCType;
        std::vector<real> uBCVal, vBCVal, wBCVal, tBCVal;

        bcparse();

        void writeParams();

    private:
        void parseYAML();
};

/**
 ********************************************************************************************************************************************
 *  \class bcparse bcparse.h "lib/io/bcparse.h"
 *  \brief  Contains all the boundary conditions set by the user through the custom BCs yaml file
 *
 *  The class parses the customBCs.yaml file and stores all the boundary condition paramters in publicly accessible constants.
 *  The class is best initialized as a constant to prevent inadvertent tampering of the global variables it contains.
 ********************************************************************************************************************************************
 */

#endif
