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

// Structs for kernels go here
struct CalcAlphaBasic {};

template <class DeviceType> class PairMTPKokkos : public PairMTP {
 public:
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

  KOKKOS_INLINE_FUNCTION
  void operator()(
      CalcAlphaBasic,
      const typename Kokkos::TeamPolicy<DeviceType, CalcAlphaBasic>::member_type &team) const;

 protected:
  int chunk_size;    // Needed to process the computation in batches to avoid running out of VRAM.

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
  // TODO: DOUBLE CHECK THE LAYOUTS!!!!
  // Alphas indicies
  Kokkos::View<int **, DeviceType> d_alpha_index_basic;    // For constructing the basic alphas.
  Kokkos::View<int **, DeviceType> d_alpha_index_times;    // For combining alphas
  Kokkos::View<int, DeviceType> d_alpha_moment_mapping;    // Maps alphas to the basis functions

  // The learned coefficients. These should probably be scatterviews but the current implementation simply uses atomics if needed.
  Kokkos::View<double *, DeviceType>
      d_radial_coeffs;    // The radial components. These specifically might benefiti from RandomAccess Trait
  Kokkos::View<double *, DeviceType> d_species_coeffs;    // The species-based constants
  Kokkos::View<double *, DeviceType> d_linear_coeffs;     // Basis coeffs

  // Global working buffers. These need to be scatterviews for atomic access.
  Kokkos::View<double ****, DeviceType> d_moment_jacobian;
  Kokkos::View<double **, DeviceType> d_moment_tensor_vals;
  Kokkos::View<double **, DeviceType> d_nbh_energy_ders_wrt_moments;

  // Typedefs for shared memory
  typedef Kokkos::View<F_FLOAT **[3], DeviceType,
                       Kokkos::DefaultExecutionSpace::scratch_memory_space,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      shared_double_3d;    // Used for coord powers
  typedef Kokkos::View<F_FLOAT **, DeviceType, Kokkos::DefaultExecutionSpace::scratch_memory_space,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      shared_double_2d;    // Used for radial basis vals, ders, and dist powers

  typedef Kokkos::DualView<F_FLOAT **, DeviceType> tdual_fparams;
  tdual_fparams k_cutsq;    // cutoffs

  int need_dup;

  using KKDeviceType = typename KKDevice<DeviceType>::value;

  template <typename DataType, typename Layout>
  using DupScatterView =
      KKScatterView<DataType, Layout, KKDeviceType, KKScatterSum, KKScatterDuplicated>;

  template <typename DataType, typename Layout>
  using NonDupScatterView =
      KKScatterView<DataType, Layout, KKDeviceType, KKScatterSum, KKScatterNonDuplicated>;

  DupScatterView<F_FLOAT *[3], typename DAT::t_f_array::array_layout> dup_f;
  DupScatterView<F_FLOAT *[6], typename DAT::t_virial_array::array_layout> dup_vatom;

  NonDupScatterView<F_FLOAT *[3], typename DAT::t_f_array::array_layout> ndup_f;
  NonDupScatterView<F_FLOAT *[6], typename DAT::t_virial_array::array_layout> ndup_vatom;

  friend void pair_virial_fdotr_compute<PairMTPKokkos>(PairMTPKokkos *);
};

}    // namespace LAMMPS_NS

#endif
#endif