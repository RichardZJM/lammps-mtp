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

#ifndef LMP_MTP_RADIAL_BASIS_H
#define LMP_MTP_RADIAL_BASIS_H

#include "pointers.h"

namespace LAMMPS_NS {

class TextFileReader;

class RadialMTPBasis : protected Pointers {
 public:
  RadialMTPBasis(TextFileReader &tfr, LAMMPS *lmp);
  RadialMTPBasis(int size, LAMMPS *lmp);
  virtual ~RadialMTPBasis();

  virtual void calc_radial_basis(double dist) = 0;
  virtual void calc_radial_basis_ders(double dist) = 0;
  virtual void calc_radial_basis(double dist, int t1, int t2);
  virtual void calc_radial_basis_ders(double dist, int t1, int t2);

  int size;             // The size of the radial basis functions
  double min_cutoff;    //  Minimum radius value
  double max_cutoff;    // Cutoff radius
  double scaling;       // All radial functions are multiplied by scaling

  // Values and derivatives for radial basis functions
  double *radial_basis_vals;
  double *radial_basis_ders;

 protected:
 private:
  //Specifically reads the basis properties (ie. cutoffs and size) and not the radial parameters
  void read_basis_properties(TextFileReader &tfr);
};

class RBChebyshev : public RadialMTPBasis {
 public:
  using RadialMTPBasis::calc_radial_basis;
  using RadialMTPBasis::calc_radial_basis_ders;
  RBChebyshev(int size, LAMMPS *lmp) : RadialMTPBasis(size, lmp) {}
  RBChebyshev(TextFileReader &tfr, LAMMPS *lmp) : RadialMTPBasis(tfr, lmp) {}
  void calc_radial_basis(double val) override;
  void calc_radial_basis_ders(double val) override;
};

class LRBSChebyshev : public RadialMTPBasis {
 public:
  LRBSChebyshev(TextFileReader &tfr, LAMMPS *lmp, int species_count);
  LRBSChebyshev(int size, int species_count, LAMMPS *lmp);
  ~LRBSChebyshev() override;

  void calc_radial_basis(double dist) override;
  void calc_radial_basis_ders(double dist) override;
  void calc_radial_basis(double dist, int t1, int t2) override;
  void calc_radial_basis_ders(double dist, int t1, int t2) override;

  int species_count;
  double *min_vals;            // Per-species-pair min cutoffs (species_count^2 flat array)
  double *max_vals;            // Per-species-pair max cutoffs (species_count^2 flat array)
  double *switching_points;    // Per-species-pair switching points (species_count^2 flat array)

  // === INSERT THESE LINE FIELDS ===
  double *inv_left_ranges;     // 1.0 / (peak - min)
  double *inv_right_ranges;    // 1.0 / (max - peak)
  double *cheb_mults;          // 2.0 / (max - min)
  double *cheb_offsets;        // (min + max) / (max - min)

  bool precomputed;               // Flag for lazy initialization of precomputed arrays
  void init_precomputations();    // Method to precompute the ranges
};

}    // namespace LAMMPS_NS

#endif    // LMP_MTP_RADIAL_BASIS_H