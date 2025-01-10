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

#ifdef PAIR_CLASS
// clang-format off
PairStyle(mtp,PairMTP);
// clang-format on
#else

#ifndef LMP_PAIR_MTP_H
#define LMP_PAIR_MTP_H

#include "pair.h"

namespace LAMMPS_NS {

class PairMTP : public Pair {
 public:
  PairMTP(class LAMMPS *);
  ~PairMTP() override;
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;

 protected:
  //   virtual void allocate();
  void read_file(char *);
  std::string potential_name = "Untitled";    //An optional name which isn't currently used.
  std::string potential_tag = "";    //An optional tag/description which isn't currently used.

  int species_count;

  double scaling = 1;
  double rcutmax;
  double rcutmin;

  double *moment_tensor_vals;
  double *mtp_basis_vals;
  double *mtp_basis_ders;
  // Needed for calculating forces
  double ***moment_jacobian;
  double **nbh_energy_ders;
  double *nbh_energy_ders_wrt_moments;

  RadialMTPBasis *radial_basis;
  double *radial_basis_coeffs;
  int radial_func_count;

  double *linear_coeffs;     // These are the moment tensor basis coeffs (eps)
  double *species_coeffs;    // For the species coefficients (0th rank moment tensor)
  int alpha_moment_count, alpha_index_basic_count, alpha_index_times_count, alpha_scalar_count,
      max_alpha_index_basic;
  int (*alpha_index_basic)[4];
  int (*alpha_index_times)[4];
  int *alpha_moment_mapping;

  //Working buffer
  double *dist_powers;
  double **coord_powers;
};

}    // namespace LAMMPS_NS

/* the definition of the PairBornGauss class (see below) is inserted here */

#endif
#endif
