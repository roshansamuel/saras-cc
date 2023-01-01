#!/bin/bash

#############################################################################################################################################
 # Saras
 # 
 # Copyright (C) 2019, Mahendra K. Verma
 #
 # All rights reserved.
 # 
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions are met:
 #     1. Redistributions of source code must retain the above copyright
 #        notice, this list of conditions and the following disclaimer.
 #     2. Redistributions in binary form must reproduce the above copyright
 #        notice, this list of conditions and the following disclaimer in the
 #        documentation and/or other materials provided with the distribution.
 #     3. Neither the name of the copyright holder nor the
 #        names of its contributors may be used to endorse or promote products
 #        derived from this software without specific prior written permission.
 # 
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 # ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 # WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 # DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
 # ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 # (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 # LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 # ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 # SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 #
 ############################################################################################################################################
 ##
 ##! \file compileSaras.sh
 #
 #   \brief Shell script to automatically compile and run SARAS
 #
 #   \author Roshan Samuel
 #   \date Jan 2020
 #   \copyright New BSD License
 #
 ############################################################################################################################################
 ##

# USER-SET PARAMETERS

# Enable PLANAR for 2D simulations
PLANAR=false

# Enable LEGACY_YAML if the version of yaml-cpp is old (for TARANG compatibility)
LEGACY_YAML=false

# Enable POST_RUN for post-processing runs
POST_RUN=false

# Enable SINGLE_PRECISION for computing with single-precision floating-point numbers
SINGLE_PRECISION=false

# Enable EXECUTE_AFTER_COMPILE if the saras executable must be run automatically after building
EXECUTE_AFTER_COMPILE=true

# If EXECUTE_AFTER_COMPILE is enabled, set number of processors for the run
PROC=8

########## NO USER-MODIFICATIONS NECESSARY BELOW THIS LINE ##########

# REMOVE PRE-EXISTING EXECUTATBLES
rm -f ../saras
rm -f ../saras_post

# IF build DIRECTORY DOESN'T EXIST, CREATE IT
if [ ! -d build ]; then
    mkdir build
fi

# SWITCH TO build DIRECTORY
cd build

# RUN Cmake WITH NECESSARY FLAGS AS SET BY USER
# Build command string
cStr="CC=mpicc CXX=mpicxx cmake ../../"
if [ "$PLANAR" = true ]; then
    cStr="$cStr -DPLANAR=ON"
fi
if [ "$SINGLE_PRECISION" = true ]; then
    cStr="$cStr -DREAL_SINGLE=ON"
fi
if [ "$LEGACY_YAML" = true ]; then
    cStr="$cStr -DYAML_LEGACY=ON"
fi
if [ "$POST_RUN" = true ]; then
    cStr="$cStr -DPOST_RUN=ON"
fi

# Execute command string
echo $cStr
eval $cStr

# COMPILE
make -j16

# SWITCH TO PARENT DIRECTORY
cd ../../

# RUN CODE IF REQUESTED BY USER
if [ "$EXECUTE_AFTER_COMPILE" = true ]; then
    echo "localhost slots="$PROC > hostfile
    if [ "$POST_RUN" = true ]; then
        mpirun --hostfile hostfile -np $PROC ./saras_post
    else
        mpirun --hostfile hostfile -np $PROC ./saras
    fi
    rm hostfile
fi
