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
/*! \file vfield.h
 *
 *  \brief Class declaration of vfield - vector field
 *
 *  \author Roshan Samuel, Ali Asad
 *  \date Nov 2019
 *  \copyright New BSD License
 *
 ********************************************************************************************************************************************
 */

#ifndef VFIELD_H
#define VFIELD_H

#include "field.h"
#include "boundary.h"
#include "derivative.h"

// Forward declarations of relevant classes
class plainsf;
class plainvf;
class sfield;
class force;

class vfield {
    private:
        // Enables Admiral Snoke to sway you over to the dark side
        bool firstOrder;

        // Diffusion coefficient - used for Pe calculation in upwindNLin()
        real diffCoeff;

        // Coefficients to adjust strength of upwinding in computeNLin()
        real omega, a, b, c, d;

        // Coefficients for finite-difference stencils in computeNLin()
        real i2dx, i2dy, i2dz;

        // Boolean flags for first and last flags along each direction.
        // These are used when upwinding is enabled for computeNlin()
        bool xfr, xlr, yfr, ylr, zfr, zlr;

        const grid &gridData;

        blitz::Array<real, 3> tempXX, tempZX, tempZZ;
#ifndef PLANAR
        blitz::Array<real, 3> tempXY, tempYZ, tempYY;
#endif

        blitz::RectDomain<3> core;

        // Average grid spacing in physical plane - used in CFL condition.
        // This will be same as hx, hy and hz for uniform grid.
        real avgDx, avgDy, avgDz;

        void upwindNLin(const vfield &V, plainvf &H);
        void morinishiNLin(const vfield &V, plainvf &H);

    public:
        field Vx, Vy, Vz;

        /** derVx, derVy and derVz are three instances of the derivative class used to compute derivatives.
         *  They correspond to finite-differencing operations along X, Y and Z directions respectively. */
        //@{
        derivative derVx, derVy, derVz;
        //@}

        /** This string is used to identify the vector field, and is useful in file-writing */
        std::string fieldName;

        /** Instance of force class to handle vector field forcing */
        force *vForcing;

        /** Instances of the \ref boundary class to impose boundary conditions on all the 6 walls for the 3 components of the vector field. */
        //@{
        boundary *uLft, *uRgt, *uFrn, *uBak, *uTop, *uBot;
        boundary *vLft, *vRgt, *vFrn, *vBak, *vTop, *vBot;
        boundary *wLft, *wRgt, *wFrn, *wBak, *wTop, *wBot;
        //@}

        vfield(const grid &gridData, std::string fieldName);

        void computeDiff(plainvf &H);
        void computeTStp(real &dt);
        void computeNLin(const vfield &V, plainvf &H);

        void divergence(plainsf &divV);

        void imposeVxBC();
        void imposeVyBC();
        void imposeVzBC();

        void imposeBCs();
        void setDiffCoeff(real dCoeff);

        vfield& operator += (plainvf &a);
        vfield& operator -= (plainvf &a);

        vfield& operator += (vfield &a);
        vfield& operator -= (vfield &a);

        vfield& operator *= (real a);

        void operator = (plainvf &a);
        void operator = (vfield &a);
        void operator = (real a);

/**
 ********************************************************************************************************************************************
 * \brief   Function to synchronise data across subdomain faces when performing parallel computations
 *
 *          Each of the individual field components have their own subroutine, \ref field#syncFaces "syncFaces" to send and
 *          receive data across its MPI decomposed sub-domains.
 *          This function calls the \ref field#syncFaces "syncFaces" function of its components to update the sub-domain boundary pads.
 ********************************************************************************************************************************************
 */
        inline void syncFaces() {
            Vx.syncFaces();
            Vy.syncFaces();
            Vz.syncFaces();
        }

/**
 ********************************************************************************************************************************************
 * \brief   Function to synchronise data across subdomain faces, edges and corners when performing parallel computations
 *
 *          Each of the individual field components have their own subroutine, \ref field#syncAll "syncAll" to send and
 *          receive data across its MPI decomposed sub-domains.
 *          This function calls the \ref field#syncAll "syncAll" function of its components to update the sub-domain boundary pads.
 ********************************************************************************************************************************************
 */
        inline void syncAll() {
            Vx.syncAll();
            Vy.syncAll();
            Vz.syncAll();
        }

        ~vfield() { };
};

/**
 ********************************************************************************************************************************************
 *  \class vfield vfield.h "lib/vfield.h"
 *  \brief Vector field class to store and operate on vector fields
 *
 *  The class stores vector fields in the form of three instances of the field class defined in <B>field.h</B>.
 *  The vector field is stored in such a way that the components are face-centered, with:
 *      - x-component located at the face centers along the yz-plane
 *      - y-component located at the face centers along the zx-plane
 *      - z-component located at the face centers along the xy-plane
 *
 *  The vector field is also equipped with a divergence operator, which returns a scalar field (sfield).
 *  However, this operation returns only cell-centered scalar field as output as most scalar fields are stored at
 *  cell centers.
 *  Moreover, the \f$ (\mathbf{u}.\nabla)\mathbf{v} \f$ operator is also provided as the function <B>computeNLin</B>
 ********************************************************************************************************************************************
 */

#endif
