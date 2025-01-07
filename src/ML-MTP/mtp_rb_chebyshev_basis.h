/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

//
// Contributing author, Richard Meng, Queen's University at Kingston, 22.11.24, contact@richardzjm.com
//

#ifndef LMP_MTP_RB_CHEBYSHEV_BASIS_H
#define LMP_MTP_RB_CHEBYSHEV_BASIS_H

#include <mtp_radial_basis.h>

namespace LAMMPS_NS {
class RBChebyshev : public RadialMTPBasis {
 public:
  RBChebyshev(PotentialFileReader &pfr, LAMMPS *lmp) : RadialMTPBasis(pfr, lmp) {}
  virtual void CalcRadialBasis(double val) override {};
  virtual void CalcRadialBasisDers(double val) override {};
};
}    // namespace LAMMPS_NS

#endif