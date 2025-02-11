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
// Contributing author, Richard Meng, Queen's University at Kingston, 10.02.25, contact@richardzjm.com
//

#ifdef PAIR_CLASS
// clang-format off
PairStyle(mtp/extrapolation,PairMTPExtrapolation);
// clang-format on
#else

#ifndef LMP_PAIR_MTP_EXTRAPOLATION_H
#define LMP_PAIR_MTP_EXTRAPOLATION_H

#include "pair_mtp.h"

namespace LAMMPS_NS {

class PairMTPExtrapolation : public PairMTP {
 public:
  ~PairMTPExtrapolation() override;
  void compute(int, int) override;         //Workhorse comuptation
  void settings(int, char **) override;    // Reads args from "pair_style"

 protected:
  void read_file(FILE *);    //Parsing file using LAMMPS utils

  int coeff_count;

  double **active_set;            // Current active set
  double **inverse_active_set;    // Inverse of the current active set

  //Working buffers
  double *extrapolation_grades;     // Extrapolation grades of all neighbourhoods
  double ***radial_jacobian;        // Jacobian of radial component wrt to basic moments
  double *radial_basic_ders;        // Energy ders wrt to basic moments
  double *radial_moment_ders;       //Ders of non-elemnetary moments wrt to basis moments
  double *basis_ders_wrt_coeffs;    // Candidate information vector
};

}    // namespace LAMMPS_NS

#endif
#endif
