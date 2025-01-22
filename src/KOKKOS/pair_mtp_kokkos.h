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
// Contributing author, Richard Meng, Queen's University at Kingston, 21.01.24, contact@richardzjm.com
//

#ifdef PAIR_CLASS
// clang-format off
PairStyle(mtp/kk,PairMTPKokkos<LMPDeviceType>);
PairStyle(mtp/kk/device,PairMTPKokkos<LMPDeviceType>);
PairStyle(mtp/kk/host,PairMTPKokkos<LMPHostType>);
// clang-format on
#else

#ifndef LMP_PAIR_MTP_KOKKOS_H
#define LMP_PAIR_MTP_KOKKOS_H

#include "kokkos_type.h"
#include "neigh_list_kokkos.h"
#include "pair_kokkos.h"
#include "pair_mtp.h"

namespace LAMMPS_NS {

template <class DeviceType> class PairMTPKokkos : public PairMTP {
 public:
  // Structs fro computation go here

  enum { EnabledNeighFlags = HALF | HALFTHREAD };
  enum { COUL_FLAG = 0 };
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  typedef EV_FLOAT value_type;

  PairMTPKokkos(class LAMMPS *);
  ~PairMTPKokkos() override;
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;

  // KOKKOS kernels go here

 protected:
  int need_dup;    // Flag to check if device duplication is needed

  // Characteric flag
  int inum, max_neighs, chunk_size, chunk_offset;
  int host_flag, neighflag;

  int eflag, vflag;    // Energy and virial flag

  typename AT::t_neighbors_2d d_neighbors;    //
  typename AT::t_int_1d_randomread d_ilist;
  typename AT::t_int_1d_randomread d_numneigh;

  DAT::tdual_efloat_1d k_eatom;
  DAT::tdual_virial_array k_vatom;
  typename AT::t_efloat_1d d_eatom;
  typename AT::t_virial_array d_vatom;

  typename AT::t_x_array_randomread x;
  typename AT::t_f_array f;
  typename AT::t_int_1d_randomread type;

  // ---------- Device Arrays  ----------
  // Alphas first
  Kokkos::View<T_INT **, DeviceType> d_alpha_index_basic;    // For constructing the basic alphas
  Kokkos::View<T_INT **, DeviceType> d_alpha_index_times;    // For combining alphas
  Kokkos::View<T_INT, DeviceType> d_alpha_moment_mapping;    // Maps alphas to the basis functions

  // The learned coefficients
  Kokkos::View<real_type *, DeviceType> d_radial_coeffs;     // The radial components
  Kokkos::View<real_type *, DeviceType> d_species_coeffs;    // The species-based constants
  Kokkos::View<real_type *, DeviceType> d_linear_coeffs;     // Basis coeffs

  typedef Kokkos::DualView<F_FLOAT **, DeviceType> tdual_fparams;
  tdual_fparams k_cutsq;    // cutoffs

  using KKDeviceType = typename KKDevice<DeviceType>::value;
  template <typename DataType, typename Layout>
  using DupScatterView =
      KKScatterView<DataType, Layout, KKDeviceType, KKScatterSum, KKScatterDuplicated>;
};

}    // namespace LAMMPS_NS

#endif
#endif