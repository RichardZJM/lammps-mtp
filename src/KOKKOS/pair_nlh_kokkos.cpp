// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing authors: Stan Moore (SNL) - ZBL Kokkos template
                         adapted for NLH
------------------------------------------------------------------------- */

#include "pair_nlh_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "error.h"
#include "force.h"
#include "kokkos.h"
#include "memory_kokkos.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "respa.h"
#include "update.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairNLHKokkos<DeviceType>::PairNLHKokkos(LAMMPS *lmp) : PairNLH(lmp)
{
  respa_enable = 0;

  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = X_MASK | F_MASK | TYPE_MASK | ENERGY_MASK | VIRIAL_MASK;
  datamask_modify = F_MASK | ENERGY_MASK | VIRIAL_MASK;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairNLHKokkos<DeviceType>::~PairNLHKokkos()
{
  if (copymode) return;

  if (allocated) {
    memoryKK->destroy_kokkos(k_eatom,eatom);
    memoryKK->destroy_kokkos(k_vatom,vatom);
  }
}


/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

template<class DeviceType>
void PairNLHKokkos<DeviceType>::init_style()
{
  PairNLH::init_style();

  // error if rRESPA with inner levels

  if (update->whichflag == 1 && utils::strmatch(update->integrate_style,"^respa")) {
    int respa = 0;
    if (((Respa *) update->integrate)->level_inner >= 0) respa = 1;
    if (((Respa *) update->integrate)->level_middle >= 0) respa = 2;
    if (respa)
      error->all(FLERR,"Cannot use Kokkos pair style with rRESPA inner/middle");
  }

  // adjust neighbor list request for KOKKOS

  neighflag = lmp->kokkos->neighflag;
  auto request = neighbor->find_request(this);
  request->set_kokkos_host(std::is_same_v<DeviceType,LMPHostType> &&
                           !std::is_same_v<DeviceType,LMPDeviceType>);
  request->set_kokkos_device(std::is_same_v<DeviceType,LMPDeviceType>);
  if (neighflag == FULL) request->enable_full();
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairNLHKokkos<DeviceType>::compute(int eflag_in, int vflag_in)
{
  eflag = eflag_in;
  vflag = vflag_in;

  if (neighflag == FULL) no_virial_fdotr_compute = 1;

  ev_init(eflag,vflag,0);

  // reallocate per-atom arrays if necessary

  if (eflag_atom) {
    memoryKK->destroy_kokkos(k_eatom,eatom);
    memoryKK->create_kokkos(k_eatom,eatom,maxeatom,"pair:eatom");
    d_eatom = k_eatom.view<DeviceType>();
  }
  if (vflag_atom) {
    memoryKK->destroy_kokkos(k_vatom,vatom);
    memoryKK->create_kokkos(k_vatom,vatom,maxvatom,"pair:vatom");
    d_vatom = k_vatom.view<DeviceType>();
  }

  atomKK->sync(execution_space,datamask_read);

  x = atomKK->k_x.view<DeviceType>();
  f = atomKK->k_f.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();
  nlocal = atom->nlocal;
  nall = atom->nlocal + atom->nghost;
  newton_pair = force->newton_pair;
  special_lj[0] = static_cast<KK_FLOAT>(force->special_lj[0]);
  special_lj[1] = static_cast<KK_FLOAT>(force->special_lj[1]);
  special_lj[2] = static_cast<KK_FLOAT>(force->special_lj[2]);
  special_lj[3] = static_cast<KK_FLOAT>(force->special_lj[3]);

  k_z.sync<DeviceType>();
  k_a1.sync<DeviceType>();
  k_a2.sync<DeviceType>();
  k_a3.sync<DeviceType>();
  k_b1.sync<DeviceType>();
  k_b2.sync<DeviceType>();
  k_b3.sync<DeviceType>();
  k_r1.sync<DeviceType>();
  k_r2.sync<DeviceType>();
  k_r1sq.sync<DeviceType>();
  k_r2sq.sync<DeviceType>();
  k_zze.sync<DeviceType>();
  k_cutsq.sync<DeviceType>();

  // loop over neighbors of my atoms

  EV_FLOAT ev = pair_compute<PairNLHKokkos<DeviceType>,void >(this,(NeighListKokkos<DeviceType>*)list);

  if (eflag_global) eng_vdwl += static_cast<double>(ev.evdwl);
  if (vflag_global) {
    virial[0] += static_cast<double>(ev.v[0]);
    virial[1] += static_cast<double>(ev.v[1]);
    virial[2] += static_cast<double>(ev.v[2]);
    virial[3] += static_cast<double>(ev.v[3]);
    virial[4] += static_cast<double>(ev.v[4]);
    virial[5] += static_cast<double>(ev.v[5]);
  }

  if (eflag_atom) {
    k_eatom.template modify<DeviceType>();
    k_eatom.sync_host();
  }

  if (vflag_atom) {
    k_vatom.template modify<DeviceType>();
    k_vatom.sync_host();
  }

  if (vflag_fdotr) pair_virial_fdotr_compute(this);

  if (eflag || vflag) atomKK->modified(execution_space,datamask_modify);
  else atomKK->modified(execution_space,F_MASK);
}

template<class DeviceType>
template<bool STACKPARAMS, class Specialisation>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
KK_FLOAT PairNLHKokkos<DeviceType>::
compute_fpair(const KK_FLOAT& rsq, const int &, const int &, const int &itype, const int &jtype) const {
  const KK_FLOAT r = sqrt(rsq);
  KK_FLOAT v, dvdr;
  eval_nlh(r, itype, jtype, v, dvdr);
  return -dvdr / r;
}

template<class DeviceType>
template<bool STACKPARAMS, class Specialisation>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
KK_FLOAT PairNLHKokkos<DeviceType>::
compute_evdwl(const KK_FLOAT &rsq, const int &, const int &, const int &itype, const int &jtype) const {
  const KK_FLOAT r = sqrt(rsq);
  KK_FLOAT v, dvdr;
  eval_nlh(r, itype, jtype, v, dvdr);
  return v;
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

template<class DeviceType>
void PairNLHKokkos<DeviceType>::allocate()
{
  PairNLH::allocate();

  int n = atom->ntypes;

  k_z     = DAT::tdual_kkfloat_1d("pair_nlh:z  ",n+1);
  k_a1    = DAT::tdual_kkfloat_2d_dl("pair_nlh:a1",n+1,n+1);
  k_a2    = DAT::tdual_kkfloat_2d_dl("pair_nlh:a2",n+1,n+1);
  k_a3    = DAT::tdual_kkfloat_2d_dl("pair_nlh:a3",n+1,n+1);
  k_b1    = DAT::tdual_kkfloat_2d_dl("pair_nlh:b1",n+1,n+1);
  k_b2    = DAT::tdual_kkfloat_2d_dl("pair_nlh:b2",n+1,n+1);
  k_b3    = DAT::tdual_kkfloat_2d_dl("pair_nlh:b3",n+1,n+1);
  k_r1    = DAT::tdual_kkfloat_2d_dl("pair_nlh:r1",n+1,n+1);
  k_r2    = DAT::tdual_kkfloat_2d_dl("pair_nlh:r2",n+1,n+1);
  k_r1sq  = DAT::tdual_kkfloat_2d_dl("pair_nlh:r1sq",n+1,n+1);
  k_r2sq  = DAT::tdual_kkfloat_2d_dl("pair_nlh:r2sq",n+1,n+1);
  k_zze   = DAT::tdual_kkfloat_2d_dl("pair_nlh:zze",n+1,n+1);
  k_cutsq = DAT::tdual_kkfloat_2d_dl("pair_nlh:cutsq",n+1,n+1);

  d_z     = k_z.view<DeviceType>();
  d_a1    = k_a1.view<DeviceType>();
  d_a2    = k_a2.view<DeviceType>();
  d_a3    = k_a3.view<DeviceType>();
  d_b1    = k_b1.view<DeviceType>();
  d_b2    = k_b2.view<DeviceType>();
  d_b3    = k_b3.view<DeviceType>();
  d_r1    = k_r1.view<DeviceType>();
  d_r2    = k_r2.view<DeviceType>();
  d_r1sq  = k_r1sq.view<DeviceType>();
  d_r2sq  = k_r2sq.view<DeviceType>();
  d_zze   = k_zze.view<DeviceType>();
  d_cutsq = k_cutsq.view<DeviceType>();
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

template<class DeviceType>
double PairNLHKokkos<DeviceType>::init_one(int i, int j)
{
  double cutone = PairNLH::init_one(i,j);

  k_z.view_host()(i) = static_cast<KK_FLOAT>(z[i]);
  k_z.view_host()(j) = static_cast<KK_FLOAT>(z[j]);
  k_a1.view_host()(i,j) = k_a1.view_host()(j,i) = static_cast<KK_FLOAT>(a1[i][j]);
  k_a2.view_host()(i,j) = k_a2.view_host()(j,i) = static_cast<KK_FLOAT>(a2[i][j]);
  k_a3.view_host()(i,j) = k_a3.view_host()(j,i) = static_cast<KK_FLOAT>(a3[i][j]);
  k_b1.view_host()(i,j) = k_b1.view_host()(j,i) = static_cast<KK_FLOAT>(b1[i][j]);
  k_b2.view_host()(i,j) = k_b2.view_host()(j,i) = static_cast<KK_FLOAT>(b2[i][j]);
  k_b3.view_host()(i,j) = k_b3.view_host()(j,i) = static_cast<KK_FLOAT>(b3[i][j]);
  k_r1.view_host()(i,j) = k_r1.view_host()(j,i) = static_cast<KK_FLOAT>(r1[i][j]);
  k_r2.view_host()(i,j) = k_r2.view_host()(j,i) = static_cast<KK_FLOAT>(r2[i][j]);
  k_r1sq.view_host()(i,j) = k_r1sq.view_host()(j,i) = static_cast<KK_FLOAT>(r1sq[i][j]);
  k_r2sq.view_host()(i,j) = k_r2sq.view_host()(j,i) = static_cast<KK_FLOAT>(r2sq[i][j]);
  k_zze.view_host()(i,j) = k_zze.view_host()(j,i) = static_cast<KK_FLOAT>(zze[i][j]);
  k_cutsq.view_host()(i,j) = k_cutsq.view_host()(j,i) = static_cast<KK_FLOAT>(cutone*cutone);

  k_z.modify_host();
  k_a1.modify_host();
  k_a2.modify_host();
  k_a3.modify_host();
  k_b1.modify_host();
  k_b2.modify_host();
  k_b3.modify_host();
  k_r1.modify_host();
  k_r2.modify_host();
  k_r1sq.modify_host();
  k_r2sq.modify_host();
  k_zze.modify_host();
  k_cutsq.modify_host();

  if (i<MAX_TYPES_STACKPARAMS+1 && j<MAX_TYPES_STACKPARAMS+1) {
    m_cutsq[i][j] = m_cutsq[j][i] = static_cast<KK_FLOAT>(cutone*cutone);
  }

  return cutone;
}

/* ----------------------------------------------------------------------
   evaluate energy and force derivative analytically
------------------------------------------------------------------------- */

template<class DeviceType>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void PairNLHKokkos<DeviceType>::eval_nlh(KK_FLOAT r, int i, int j, KK_FLOAT &v, KK_FLOAT &dvdr) const {

  const KK_FLOAT rinv = static_cast<KK_FLOAT>(1.0) / r;
  const KK_FLOAT e1 = exp(-d_b1(i,j)*r);
  const KK_FLOAT e2 = exp(-d_b2(i,j)*r);
  const KK_FLOAT e3 = exp(-d_b3(i,j)*r);

  const KK_FLOAT sum = d_a1(i,j)*e1 + d_a2(i,j)*e2 + d_a3(i,j)*e3;
  const KK_FLOAT sum_p = -d_a1(i,j)*d_b1(i,j)*e1
                       - d_a2(i,j)*d_b2(i,j)*e2
                       - d_a3(i,j)*d_b3(i,j)*e3;

  const KK_FLOAT zzeij = d_zze(i,j);
  const KK_FLOAT E = zzeij*sum*rinv;
  const KK_FLOAT dEdr = zzeij*(sum_p - sum*rinv)*rinv;

  const KK_FLOAT r1_ij = d_r1(i,j);
  const KK_FLOAT r2_ij = d_r2(i,j);

  if (r <= r1_ij) {
    v = E;
    dvdr = dEdr;
  } else if (r < r2_ij) {
    const KK_FLOAT w = static_cast<KK_FLOAT>(1.0) / (r2_ij - r1_ij);
    const KK_FLOAT x = (r - r1_ij) * w;

    // Smooth 5th-order polynomial switching function
    const KK_FLOAT f = static_cast<KK_FLOAT>(1.0) -
      x*x*x*(static_cast<KK_FLOAT>(6.0)*x*x -
             static_cast<KK_FLOAT>(15.0)*x +
             static_cast<KK_FLOAT>(10.0));
    const KK_FLOAT dfdr = -static_cast<KK_FLOAT>(30.0)*w*x*x*
      (x-static_cast<KK_FLOAT>(1.0))*(x-static_cast<KK_FLOAT>(1.0));

    v = E * f;
    dvdr = dEdr * f + E * dfdr;
  } else {
    v = static_cast<KK_FLOAT>(0.0);
    dvdr = static_cast<KK_FLOAT>(0.0);
  }
}

namespace LAMMPS_NS {
template class PairNLHKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PairNLHKokkos<LMPHostType>;
#endif
}