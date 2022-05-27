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
/*! \file les.h
 *
 *  \brief Class declaration of LES Modules
 *
 *  \author Roshan Samuel
 *  \date Sep 2020
 *  \copyright New BSD License
 *
 ********************************************************************************************************************************************
 */

/********************************************************************************************************************************************
 *
 * The stretched-vortex LES model defined by the derived class, spiral, is adapted
 * from the code provided by Dale I Pullin at Caltech.
 * A detailed description of the module can be found in doc/spiral.pdf
 *
 ********************************************************************************************************************************************
 */

#ifndef LES_H
#define LES_H

#include "grid.h"
#include "vfield.h"
#include "sfield.h"
#include "plainvf.h"
#include "plainsf.h"

class les {
    public:
        les(const grid &mesh);

        real totalSGKE, totalDisp, totalNuSG;

        virtual void computeSG(plainvf &nseRHS, vfield &V);
        virtual void computeSG(plainvf &nseRHS, plainsf &tmpRHS, vfield &V, sfield &T);

    protected:
        const grid &mesh;

        blitz::RectDomain<3> core;
};

/**
 ********************************************************************************************************************************************
 *  \class les les.h "lib/les/les.h"
 *  \brief Contains all the global variables related to the LES models used by SARAS
 *
 ********************************************************************************************************************************************
 */

class wallModel {
    public:
        wallModel(const grid &mesh, const int bcWall, const real &kDiff);

        /** The flag is true for MPI ranks on which the wall-model has to be applied. */
        bool rankFlag;

        /** The const integer denotes the wall at which the wall-model is being applied. */
        const int wallNum;

        /** The index of the near wall data point which is read to compute the slip velocity. */
        int wInd;

        /** Distance of actual no-slip wall from the virtual wall and distance of first mesh point */
        real h0, bc_h;

        blitz::Array<real, 2> eta0, K0, q;
        blitz::Array<real, 2> Tii, Tjj, Tij;
        blitz::Array<real, 2> bcU, bcV, bcW;
        blitz::Array<real, 2> vi, vj, vii, vjj, vij;

        real updateK0(real K, real Tik, real Tjk);
        void advanceEta0(vfield &V, sfield &P, real gamma, real zeta);

    private:
        const grid &mesh;

        // Kinematic viscosity
        const real &nu;

        /** Emprical constant denoting denoting non-dimensional thickness of viscous layer. */
        real h_nu_plus;

        /** Denotes the dimension normal to the wall at which the wall-model is applied. */
        int shiftDim;

        /** Direction along which the wall slice will be shifted when applying wall-model. */
        int shiftVal;

        /** TinyVectors that denote the lower bound, upper bound and size of the virtual wall slice. */
        blitz::TinyVector<int, 2> dlBnd, duBnd, dSize;

        blitz::Array<real, 2> eta0temp;

        void computeBCVel();

        inline real uTau2u(real uTau, real dynKarm);
};

/**
 ********************************************************************************************************************************************
 *  \class wallModel les.h "lib/les/les.h"
 *  \brief An additional object that can be used with the spiral LES model when invoking
 *  the wall-model for WMLES simulations.
 *
 ********************************************************************************************************************************************
 */

class spiral: public les {
    public:
        bool sgfFlag;

        spiral(const grid &mesh, const real &kDiff);
        spiral(const grid &mesh, const real &kDiff, std::vector<wallModel*> &wmList);

        void computeSG(plainvf &nseRHS, vfield &V);
        void computeSG(plainvf &nseRHS, plainsf &tmpRHS, vfield &V, sfield &T);

        ~spiral();

    private:
        // Kinematic viscosity
        const real &nu;

        // Array limits for loops
        int xS, xE, yS, yE, zS, zE;

        // Temporary variables to store output from spiral LES solver
        real sTxx, sTyy, sTzz, sTxy, sTyz, sTzx;

        // Temporary variables to store the components of the strain rate tensor
        real Sxx, Syy, Szz, Sxy, Syz, Szx;

        // Cutoff wavelength
        real del;

        // Sub-grid energy
        real K;

        // Vector of wall-model objects to be used if wall-model is enabled
        std::vector<wallModel*> wmList;

        // These 9 arrays store components of the velocity gradient tensor intially
        // Then they are reused to store the derivatives of stress tensor to calculate its divergence
        blitz::Array<real, 3> A11, A12, A13;
        blitz::Array<real, 3> A21, A22, A23;
        blitz::Array<real, 3> A31, A32, A33;

        // These 3 arrays are used only when computing scalar turbulent SGS diffusion
        blitz::Array<real, 3> B1, B2, B3;

        // These are three 3x3x3 arrays containing local interpolated velocities
        // These are used to calculate the structure function within the spiral les routine
        blitz::Array<real, 3> u, v, w;

        // The alignment vector of the sub-grid spiral vortex
        blitz::TinyVector<real, 3> e;

        // These three tiny vectors hold the x, y and z coordinates of the 3x3x3 cubic cell over which
        // the structure function will be computed
        blitz::TinyVector<real, 3> x, y, z;

        // The following 3x3 matrix stores the velocity gradient tensor
        blitz::Array<real, 2> dudx;

        // The following 3x1 vector stores the temperature gradient vector
        blitz::TinyVector<real, 3> dsdx;

        // These 3 scalar fields hold the sub-grid scalar flux vector
        sfield *qX, *qY, *qZ;

        // These scalar fields are basically the velocity fields interpolated to cell-centers
        sfield *Vxcc, *Vycc, *Vzcc;

        // These 6 scalar fields hold the sub-grid stress tensor field
        sfield *Txx, *Tyy, *Tzz, *Txy, *Tyz, *Tzx;

        void sgsStress(real *Txx, real *Tyy, real *Tzz,
                       real *Txy, real *Tyz, real *Tzx);

        void sgsFlux(real *qx, real *qy, real *qz);

        real keIntegral(real k);

        real sfIntegral(real d);

        real eigenvalueSymm();

        blitz::TinyVector<real, 3> eigenvectorSymm(real eigval);

        void updateWMArrays(vfield &V);

        void updateWMk0(int iX, int iY, int iZ);
};

/**
 ********************************************************************************************************************************************
 *  \class spiral les.h "lib/les/les.h"
 *  \brief The derived class from les to implement the stretched spiral vortex LES model
 *
 ********************************************************************************************************************************************
 */

#endif
