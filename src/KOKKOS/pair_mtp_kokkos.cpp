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

#include "pair_mtp_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "kokkos.h"
#include "math_const.h"
#include "memory_kokkos.h"
#include "neigh_request.h"
#include "neighbor_kokkos.h"

using namespace LAMMPS_NS;
using namespace MathConst;

/* ---------------------------------------------------------------------- */

template <class DeviceType> PairMTPKokkos<DeviceType>::PairMTPKokkos(LAMMPS(*lmp)) : PairMTP(lmp)
{
  respa_enable = 0;

  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = EMPTY_MASK;
  datamask_modify = EMPTY_MASK;

  host_flag = (execution_space == Host);
}

/* ---------------------------------------------------------------------- */

template <class DeviceType> PairMTPKokkos<DeviceType>::~PairMTPKokkos()
{
  if (copymode) return;

  memoryKK->destroy_kokkos(k_eatom, eatom);
  memoryKK->destroy_kokkos(k_vatom, vatom);
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

template <class DeviceType> void PairMTPKokkos<DeviceType>::init_style()
{
  if (host_flag) {
    if (lmp->kokkos->nthreads > 1)
      error->all(FLERR,
                 "Pair style mtp/kk can currently only run on a single."
                 "CPU thread");

    PairPACE::init_style();
    return;
  }

  if (force->newton_pair == 0) error->all(FLERR, "Pair style MTP requires newton pair on");

  // neighbor list request for KOKKOS
  neighflag = lmp->kokkos->neighflag;

  auto request = neighbor->add_request(this, NeighConst::REQ_FULL);
  request->set_kokkos_host(std::is_same_v<DeviceType, LMPHostType> &&
                           !std::is_same_v<DeviceType, LMPDeviceType>);
  request->set_kokkos_device(std::is_same_v<DeviceType, LMPDeviceType>);
  if (neighflag == FULL) error->all(FLERR, "Must use half neighbor list style with pair pace/kk");
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

template <class DeviceType> double PairMTPKokkos<DeviceType>::init_one(int i, int j)
{
  double cutone = PairMTP::init_one(i, j);

  k_cutsq.h_view(i, j) = k_cutsq.h_view(j, i) = cutone * cutone;
  k_cutsq.template modify<LMPHostType>();

  return cutone;
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

template <class DeviceType> void PairMTPKokkos<DeviceType>::coeff(int narg, char **arg)
{
  PairMTP::coeff(narg, arg);
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

template <class DeviceType> void PairMTPKokkos<DeviceType>::settings(int narg, char **arg)
{
  PairMTP::coeff(
      narg, arg);    // This also calls the read files which parse and loads the necessary arrays

  // ---------- Now we move arrays to device ----------

  // First we set up the index lists
  MemKK::realloc_kokkos(d_alpha_index_basic, "mtp:alpha_index_basic", alpha_index_basic_count, 4);
  MemKK::realloc_kokkos(d_alpha_index_times, "mtp:alpha_index_times", alpha_index_times_count, 4);
  MemKK::realloc_kokkos(d_alpha_moment_mapping, "mtp:moment_mapping", alpha_scalar_count);

  // Setup the learned coefficients
  int radial_coeff_count = species_count * species_coeffs * radial_basis_size * radial_func_count;
  MemKK::realloc_kokkos(d_radial_coeffs, "mtp:radial_coeffs", radial_coeff_count);
  MemKK::realloc_kokkos(d_species_coeffs, "mtp:species_coeffs", species_count);
  MemKK::realloc_kokkos(d_linear_coeffs, "mtp:linear_coeffs", alpha_scalar_count);

  //Declare host arrays
  auto h_alpha_index_basic = Kokkos::create_mirror_view(d_alpha_index_basic);
  auto h_alpha_index_times = Kokkos::create_mirror_view(d_alpha_index_times);
  auto h_alpha_moment_mapping = Kokkos::create_mirror_view(d_alpha_moment_mapping);
  auto h_radial_basis_coeffs = Kokkos::create_mirror_view(d_radial_coeffs);
  auto h_species_coeffs = Kokkos::create_mirror_view(d_species_coeffs);
  auto h_linear_coeffs = Kokkos::create_mirror_view(d_linear_coeffs);

  //Populate the host arrays
  for (int j = 0; j < 4; j++) {
    for (int i = 0; i < alpha_index_basic_count; i++)
      h_alpha_index_basic(i, j) = alpha_index_basic[i][j];
    for (int i = 0; i < alpha_index_times_count; i++)
      h_alpha_index_times(i, j) = alpha_index_times[i][j];
  }
  for (int i = 0; i < alpha_scalar_count; i++) {
    h_alpha_moment_mapping(i) = alpha_moment_mapping[i];
    h_linear_coeffs(i) = linear_coeffs[i];
  }
  for (int i = 0; i < radial_coeff_count; i++) h_radial_basis_coeffs(i) = radial_basis_coeffs[i];
  for (int i = 0; i < species_count; i++) h_species_coeffs(i) = species_coeffs[i];

  // Peform the copy from host to device
  Kokkos::deep_copy(d_alpha_index_basic, h_alpha_index_basic);
  Kokkos::deep_copy(d_alpha_index_times, h_alpha_index_times);
  Kokkos::deep_copy(d_alpha_moment_mapping, h_alpha_moment_mapping);
  Kokkos::deep_copy(d_radial_coeffs, h_radial_coeffs);
  Kokkos::deep_copy(d_species_coeffs, h_species_coeffs);
  Kokkos::deep_copy(d_linear_coeffs, h_linear_coeffs);
}