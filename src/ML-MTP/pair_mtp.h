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
  virtual void allocate();
  void read_files(char *, char *);
  double rcutmax;
  double rcutmin;
};

}    // namespace LAMMPS_NS

/* the definition of the PairBornGauss class (see below) is inserted here */

#endif
#endif
