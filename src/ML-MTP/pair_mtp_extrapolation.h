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
  PairMTPExtrapolation(LAMMPS *lmp) : PairMTP(lmp) {};
  ~PairMTPExtrapolation();
  void compute(int, int) override;         //Workhorse comuptation
  void settings(int, char **) override;    // Reads args from "pair_style"

 protected:
  void read_file(FILE *);    //Parsing file using LAMMPS utils

  int coeff_count;    // Sum of radial, species and linear coeff count

  bool untrained_potential = false;
  bool pool_grades;              // Is configuration mode?
  int sampling_frequency = 1;    // Sample frequency, default of 1
  double select_threshold;       // Grade threshold for selection
  double break_threshold;        // Grade threshold for termination

  // Active set
  double **active_set;            // Current active set
  double **inverse_active_set;    // Inverse of the current active set

  //Working buffers
  double ***radial_jacobian;    // Jacobian of radial component wrt to basic moments
  //   double *radial_basic_ders;            // Energy ders wrt to basic moments
  double *radial_moment_ders;        //Ders of non-elemnetary moments wrt to basis moments
  double *energy_ders_wrt_coeffs;    // Candidate information vector

  // Only needed for neigbhourhood mode
  double *extrapolation_grades = nullptr;    // Extrapolation grades of all neighbourhoods
};

}    // namespace LAMMPS_NS

#endif
#endif
