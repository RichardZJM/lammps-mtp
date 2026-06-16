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

#ifdef PAIR_CLASS
// clang-format off
PairStyle(nlh,PairNLH);
// clang-format on
#else

#ifndef LMP_PAIR_NLH_H
#define LMP_PAIR_NLH_H

#include "pair.h"

namespace LAMMPS_NS {

class PairNLH : public Pair {
 public:
  PairNLH(class LAMMPS *);
  ~PairNLH() override;
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;
  void write_restart(FILE *) override;
  void read_restart(FILE *) override;
  void write_restart_settings(FILE *) override;
  void read_restart_settings(FILE *) override;
  void write_data(FILE *) override;
  void write_data_all(FILE *) override;
  double single(int, int, int, int, double, double, double, double &) override;

 protected:
  double *z;
  double **a1, **a2, **a3;
  double **b1, **b2, **b3;
  double **r1, **r2;
  double **r1sq, **r2sq;
  double **zze;

  virtual void allocate();
  void eval_nlh(double, int, int, double &, double &);
  void set_coeff(int, int, double, double, double, double, double, double, double, double, double,
                 double);
};
}    // namespace LAMMPS_NS

#endif
#endif