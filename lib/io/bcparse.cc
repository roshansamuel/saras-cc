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
/*! \file bcparse.cc
 *
 *  \brief Definitions for functions of class bcparse
 *  \sa bcparse.h
 *  \author Roshan Samuel
 *  \date Mar 2024
 *  \copyright New BSD License
 *
 ********************************************************************************************************************************************
 */

#include <iostream>
#include "bcparse.h"
#include "mpi.h"

bcparse::bcparse() {
    parseYAML();
}

/**
 ********************************************************************************************************************************************
 * \brief   Function to open the yaml file and parse the parameters
 *
 *          The function opens the customBCs.yaml file and parses the boundary conditions into its member variables that are publicly
 *          accessible.
 ********************************************************************************************************************************************
 */
void bcparse::parseYAML() {
    std::ifstream inFile;

    int inpType;
    real inpVal;

#ifdef PLANAR
    std::vector<std::string> sList = {
        "Left Wall",
        "Right Wall",
        "Bottom Wall",
        "Top Wall"
    };
#else
    std::vector<std::string> sList = {
        "Left Wall",
        "Right Wall",
        "Front Wall",
        "Back Wall",
        "Bottom Wall",
        "Top Wall"
    };
#endif

    inFile.open("input/customBCs.yaml", std::ifstream::in);

#ifdef YAML_LEGACY
    YAML::Node yamlNode;
    YAML::Parser parser(inFile);

    parser.GetNextDocument(yamlNode);

    for (const std::string& sStr : sList) {
        yamlNode["V"]["Vx"][sStr]["BC Type"] >> inpType;
        uBCType.push_back(inpType);

        yamlNode["V"]["Vx"][sStr]["BC Value"] >> inpVal;
        uBCVal.push_back(inpVal);
    }

#ifndef PLANAR
    for (const std::string& sStr : sList) {
        yamlNode["V"]["Vy"][sStr]["BC Type"] >> inpType;
        vBCType.push_back(inpType);

        yamlNode["V"]["Vy"][sStr]["BC Value"] >> inpVal;
        vBCVal.push_back(inpVal);
    }
#endif

    for (const std::string& sStr : sList) {
        yamlNode["V"]["Vz"][sStr]["BC Type"] >> inpType;
        wBCType.push_back(inpType);

        yamlNode["V"]["Vz"][sStr]["BC Value"] >> inpVal;
        wBCVal.push_back(inpVal);
    }

    for (const std::string& sStr : sList) {
        yamlNode["T"]["F"][sStr]["BC Type"] >> inpType;
        tBCType.push_back(inpType);

        yamlNode["T"]["F"][sStr]["BC Value"] >> inpVal;
        tBCVal.push_back(inpVal);
    }
#else
    YAML::Node yamlNode = YAML::Load(inFile);

    for (const std::string& sStr : sList) {
        inpType = yamlNode["V"]["Vx"][sStr]["BC Type"].as<int>();
        uBCType.push_back(inpType);

        inpVal = yamlNode["V"]["Vx"][sStr]["BC Value"].as<real>();
        uBCVal.push_back(inpVal);
    }

#ifndef PLANAR
    for (const std::string& sStr : sList) {
        inpType = yamlNode["V"]["Vy"][sStr]["BC Type"].as<int>();
        vBCType.push_back(inpType);

        inpVal = yamlNode["V"]["Vy"][sStr]["BC Value"].as<real>();
        vBCVal.push_back(inpVal);
    }
#endif

    for (const std::string& sStr : sList) {
        inpType = yamlNode["V"]["Vz"][sStr]["BC Type"].as<int>();
        wBCType.push_back(inpType);

        inpVal = yamlNode["V"]["Vz"][sStr]["BC Value"].as<real>();
        wBCVal.push_back(inpVal);
    }

    for (const std::string& sStr : sList) {
        inpType = yamlNode["T"]["F"][sStr]["BC Type"].as<int>();
        tBCType.push_back(inpType);

        inpVal = yamlNode["T"]["F"][sStr]["BC Value"].as<real>();
        tBCVal.push_back(inpVal);
    }
#endif

    inFile.close();
}

/**
 ********************************************************************************************************************************************
 * \brief   Function to write all the BC parameter values to I/O
 *
 *          All the user set parameters have to be written to I/O so that the runlog or any out file
 *          contains all the relevant information about the case that was run.
 *          This public function has to be called from the solver by one rank only.
 *          Additionally it is called only if it has been specified to use custom BCs.
 ********************************************************************************************************************************************
 */
void bcparse::writeParams() {
    std::cout << std::endl << "Writing BC parameters from the YAML input file for reference" << std::endl << std::endl;
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

