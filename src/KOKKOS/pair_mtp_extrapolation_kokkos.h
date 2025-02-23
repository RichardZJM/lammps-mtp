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
PairStyle(mtp/extrapolation/kk,PairMTPExtrapolationKokkos<LMPDeviceType>);
PairStyle(mtp/extrapolation/kk/device,PairMTPExtrapolationKokkos<LMPDeviceType>);
PairStyle(mtp/extrapolation/kk/host,PairMTPExtrapolationKokkos<LMPHostType>);
// clang-format on
#else

#ifndef LMP_PAIR_MTP_KOKKOS_EXTRAPOLATION_H
#define LMP_PAIR_MTP_KOKKOS_EXTRAPOLATION_H

#include "kokkos_type.h"
#include "neigh_list_kokkos.h"
#include "pair_kokkos.h"
#include "pair_mtp_extrapolation.h"

namespace LAMMPS_NS {

template <class DeviceType> class PairMTPExtrapolationKokkos : public PairMTPExtrapolation {
 public:
  // Structs for kernels
  struct TagPairMTPInitMomentValsDers {};
  struct TagPairMTPInitRadJacobian {};
  struct TagPairMTPComputeAlphaBasic {};
  struct TagPairMTPComputeAlphaBasicRad {};
  struct TagPairMTPComputeAlphaTimes {};
  struct TagPairMTPSetScalarNbhDers {};
  struct TagPairMTPComputeNbhDers {};
  struct TagPairMTPReduceCoeffDers {};
  struct TagPairMTPTransferBasisDers {};
  struct TagPairMTPComputeNbhGrades {};
  struct TagPairMTPComputeCfgGrade {};
  template <int NEIGHFLAG, int EVFLAG> struct TagPairMTPComputeForce {};

  enum { EnabledNeighFlags = HALF | HALFTHREAD };
  enum { COUL_FLAG = 0 };
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  typedef EV_FLOAT value_type;

  PairMTPExtrapolationKokkos(class LAMMPS *);
  ~PairMTPExtrapolationKokkos();
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;
  void evaluate_grades();

  // ========== Kokkos kernels ==========
  //Utility routines
  template <class TagStyle> void check_team_size_for(int, int &, int);

  template <typename scratch_type>
  int scratch_size_helper(int values_per_team);    // Helps calcs scratch size for calcalphabasic

  template <int NEIGHFLAG>
  KOKKOS_INLINE_FUNCTION void v_tally_xyz(EV_FLOAT &ev, const int &i, const int &j,
                                          const F_FLOAT &fx, const F_FLOAT &fy, const F_FLOAT &fz,
                                          const F_FLOAT &delx, const F_FLOAT &dely,
                                          const F_FLOAT &delz) const;

  // ---------- MTP routines (inorder of execution) ----------

  //Kernels for initing working views
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairMTPInitMomentValsDers, const int &ii, const int &k) const;

  KOKKOS_INLINE_FUNCTION    // Only runs on steps when we sample extrapolation
      void
      operator()(TagPairMTPInitRadJacobian, const int &ii, const int &k) const;

  // Kernels for computation
  KOKKOS_INLINE_FUNCTION
  void
  operator()(TagPairMTPComputeAlphaBasic,
             const typename Kokkos::TeamPolicy<DeviceType, TagPairMTPComputeAlphaBasic>::member_type
                 &team) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(
      TagPairMTPComputeAlphaBasicRad,
      const typename Kokkos::TeamPolicy<DeviceType, TagPairMTPComputeAlphaBasicRad>::member_type
          &team) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairMTPComputeAlphaTimes, const int &ii) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairMTPSetScalarNbhDers, const int &ii, const int &k) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairMTPComputeNbhDers, const int &ii) const;

  template <int NEIGHFLAG, int EVFLAG>
  KOKKOS_INLINE_FUNCTION void
  operator()(TagPairMTPComputeForce<NEIGHFLAG, EVFLAG>,
             const int &ii) const;    // This eventually calls the below version

  template <int NEIGHFLAG, int EVFLAG>
  KOKKOS_INLINE_FUNCTION void
  operator()(TagPairMTPComputeForce<NEIGHFLAG, EVFLAG>, const int &ii,
             EV_FLOAT &) const;    // With global energy reduction as needed

  // Kernels for extrapolation computation
  KOKKOS_INLINE_FUNCTION
  void
  operator()(TagPairMTPReduceCoeffDers,
             const typename Kokkos::TeamPolicy<DeviceType, TagPairMTPReduceCoeffDers>::member_type
                 &team) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairMTPTransferBasisDers, const int &kk) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(
      TagPairMTPComputeNbhGrades,
      const typename Kokkos::TeamPolicy<DeviceType, TagPairMTPComputeNbhGrades>::member_type &team,
      F_FLOAT &nbh_max_grade) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(
      TagPairMTPComputeCfgGrade,
      const typename Kokkos::TeamPolicy<DeviceType, TagPairMTPComputeCfgGrade>::member_type &team,
      F_FLOAT &cfg_max_grade) const;

 protected:
  int chunk_size,
      chunk_offset;    // Needed to process the computation in batches to avoid running out of VRAM.

  // Characteric flags
  int inum, max_neighs;
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
  // Alphas indicies
  Kokkos::View<int **, DeviceType> d_alpha_index_basic;      // For constructing the basic alphas.
  Kokkos::View<int **, DeviceType> d_alpha_index_times;      // For combining alphas
  Kokkos::View<int *, DeviceType> d_alpha_moment_mapping;    // Maps alphas to the basis functions.

  // The learned coefficients.
  Kokkos::View<double *, DeviceType>
      d_radial_basis_coeffs;    // The radial components. These specifically might benefiti from RandomAccess Trait
  Kokkos::View<double *, DeviceType> d_species_coeffs;    // The species-based constants
  Kokkos::View<double *, DeviceType> d_linear_coeffs;     // Basis coeffs

  // Inverse active set and grades. Only needed in neigbhourhood mode.
  Kokkos::View<double **, DeviceType> d_inverse_active_set;
  Kokkos::View<double *, DeviceType> d_nbh_extrapolation_grades;

  // Source for candidate vector reduction. Only needed in configuration mode.
  // We need two arrays to account for multiple chunks.
  Kokkos::View<double *, DeviceType> d_energy_ders_wrt_coeffs;
  Kokkos::View<double *, DeviceType> d_tmp_energy_ders_wrt_coeffs;

  // Global working buffers.
  Kokkos::View<double ****, DeviceType> d_moment_jacobian;
  Kokkos::View<double ***, DeviceType> d_radial_jacobian;
  Kokkos::View<double **, DeviceType> d_moment_tensor_vals;
  Kokkos::View<double **, DeviceType> d_nbh_energy_ders_wrt_moments;
  Kokkos::View<bool **, DeviceType> d_within_cutoff;

  // Typedefs for shared memory
  typedef Kokkos::View<F_FLOAT **[3], typename DeviceType::scratch_memory_space,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      shared_double_3d;    // Used for coord powers
  typedef Kokkos::View<F_FLOAT **, typename DeviceType::scratch_memory_space,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      shared_double_2d;    // Used for radial basis vals, ders, and dist powers
  typedef Kokkos::View<F_FLOAT *, typename DeviceType::scratch_memory_space,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      shared_double_1d;    // Used for storing derivatives

  int need_dup;

  // ---------- Define the forces, per-atom energy, and virials----------
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

  friend void pair_virial_fdotr_compute<PairMTPExtrapolationKokkos>(PairMTPExtrapolationKokkos *);
};

}    // namespace LAMMPS_NS

#endif
#endif