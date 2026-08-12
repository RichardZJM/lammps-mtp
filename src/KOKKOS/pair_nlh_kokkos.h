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
PairStyle(nlh/kk,PairNLHKokkos<LMPDeviceType>);
PairStyle(nlh/kk/device,PairNLHKokkos<LMPDeviceType>);
PairStyle(nlh/kk/host,PairNLHKokkos<LMPHostType>);
// clang-format on
#else

// clang-format off
#ifndef LMP_PAIR_NLH_KOKKOS_H
#define LMP_PAIR_NLH_KOKKOS_H

#include "pair_nlh.h"
#include "pair_kokkos.h"
#include "neigh_list_kokkos.h"

namespace LAMMPS_NS {

template<class DeviceType>
class PairNLHKokkos : public PairNLH {
 public:
  enum {EnabledNeighFlags=FULL|HALFTHREAD|HALF};
  enum {COUL_FLAG=0};
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  PairNLHKokkos(class LAMMPS *);
  ~PairNLHKokkos() override;
  void compute(int, int) override;
  void init_style() override;
  double init_one(int, int) override;

 private:
  DAT::tdual_kkfloat_1d k_z;
  DAT::tdual_kkfloat_2d_dl k_a1,k_a2,k_a3,k_b1,k_b2,k_b3,k_r1,k_r2,k_r1sq,k_r2sq,k_zze;
  DAT::tdual_kkfloat_2d_dl k_cutsq;

  typename AT::t_kkfloat_1d d_z;
  typename AT::t_kkfloat_2d_dl d_a1,d_a2,d_a3,d_b1,d_b2,d_b3,d_r1,d_r2,d_r1sq,d_r2sq,d_zze;
  typename AT::t_kkfloat_2d_dl d_cutsq;

  typename AT::t_kkfloat_1d_3_lr_randomread x;
  typename AT::t_kkacc_1d_3 f;
  typename AT::t_int_1d_randomread type;

  DAT::ttransform_kkacc_1d k_eatom;
  DAT::ttransform_kkacc_1d_6 k_vatom;
  typename AT::t_kkacc_1d d_eatom;
  typename AT::t_kkacc_1d_6 d_vatom;

  KK_FLOAT m_cutsq[MAX_TYPES_STACKPARAMS+1][MAX_TYPES_STACKPARAMS+1];

  int newton_pair;
  int neighflag;
  int nlocal,nall,eflag,vflag;
  KK_FLOAT special_lj[4];

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void eval_nlh(KK_FLOAT, int, int, KK_FLOAT &, KK_FLOAT &) const;

  template<bool STACKPARAMS, class Specialisation>
// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  KK_FLOAT compute_fpair(const KK_FLOAT& rsq, const int& i, const int&j, const int& itype, const int& jtype) const;

  template<bool STACKPARAMS, class Specialisation>
// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  KK_FLOAT compute_evdwl(const KK_FLOAT& rsq, const int& i, const int&j, const int& itype, const int& jtype) const;

  template<bool STACKPARAMS, class Specialisation>
// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  KK_FLOAT compute_ecoul(const KK_FLOAT& /*rsq*/, const int& /*i*/, const int& /*j*/,
                        const int& /*itype*/, const int& /*jtype*/) const { return 0; }

  void allocate() override;

  friend struct PairComputeFunctor<PairNLHKokkos,FULL,true,0>;
  friend struct PairComputeFunctor<PairNLHKokkos,FULL,true,1>;
  friend struct PairComputeFunctor<PairNLHKokkos,HALF,true>;
  friend struct PairComputeFunctor<PairNLHKokkos,HALFTHREAD,true>;
  friend struct PairComputeFunctor<PairNLHKokkos,FULL,false,0>;
  friend struct PairComputeFunctor<PairNLHKokkos,FULL,false,1>;
  friend struct PairComputeFunctor<PairNLHKokkos,HALF,false>;
  friend struct PairComputeFunctor<PairNLHKokkos,HALFTHREAD,false>;
  friend EV_FLOAT pair_compute_neighlist<PairNLHKokkos,FULL,0>(PairNLHKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT pair_compute_neighlist<PairNLHKokkos,FULL,1>(PairNLHKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT pair_compute_neighlist<PairNLHKokkos,HALF>(PairNLHKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT pair_compute_neighlist<PairNLHKokkos,HALFTHREAD>(PairNLHKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT pair_compute<PairNLHKokkos>(PairNLHKokkos*,NeighListKokkos<DeviceType>*);
  friend void pair_virial_fdotr_compute<PairNLHKokkos>(PairNLHKokkos*);
};

}

#endif
#endif