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
  // We may need to process in chunks to deal with VRAM limitations

  if (narg != 3 || arg[1] != "chunksize")
    error->all(FLERR,
               "Pair MTP requires 3 arguments {potential_file}, \"chunksize\", {chunksize}".);

  chunksize = utils::inumeric(FLERR, arg[2], true, lmp);

  PairMTP ::settings(
      1, arg);    // This also calls read_file which parses and loads the necessary arrays

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

  //Setup the working arrays. It might be preferable for these to be scatter view
  MemKK::realloc_kokkos(d_moment_jacobian, "mtp:moment_jacobian", chunk_size,
                        alpha_index_basic_count, 100,
                        3);    // Arbitrary initial value (to be reallocated with max neighs)
  MemKK::realloc_kokkos(d_moment_tensor_vals, "mtp:moment_tensor_vals", chunk_size,
                        alpha_moment_count);
  MemKK::realloc_kokkos(d_nbh_energy_ders_wrt_moments, "mtp:nbh_energy_ders_wrt_moments",
                        chunk_size, alpha_moment_count);

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

  // No need to deep copy the working buffers.
}

/* ----------------------------------------------------------------------
   This version is a straightforward implementation
   ---------------------------------------------------------------------- */

template <class DeviceType> void PairMTPKokkos<DeviceType>::compute(int eflag_in, int vflag_in)
{
  // If we are running on host we just use the base implementation (lololol?)
  if (host_flag) {
    atomKK->sync(Host, X_MASK | F_MASK | TYPE_MASK);
    PairMTP::compute(eflag_in, vflag_in);
    atomKK->modified(Host, F_MASK);
    return;
  }

  eflag = eflag_in;
  vlag = vflag_in;

  if (neighflag == FULL)
    no_virial_fdotr_compute =
        1;    // I'm pretty sure we force it to be FULL so the if statement doesn't matter?

  ev_init(eflag, vflag, 0);

  // reallocate per-atom arrays if necessary
  if (eflag_atom) {
    memoryKK->destroy_kokkos(k_eatom, eatom);
    memoryKK->create_kokkos(k_eatom, eatom, maxeatom, "pair:eatom");
    d_eatom = k_eatom.view<DeviceType>();
  }
  if (vflag_atom) {
    memoryKK->destroy_kokkos(k_vatom, vatom);
    memoryKK->create_kokkos(k_vatom, vatom, maxvatom, "pair:vatom");
    d_vatom = k_vatom.view<DeviceType>();
  }

  copymode = 1;
  int newton_pair = force->newton_pair;
  if (newton_pair == false) error->all(FLERR, "PairMTPKokkos requires 'newton on'");

  // Now, ensure the atom data is synced
  atomKK->sync(execution_space, X_MASK | F_MASK | TYPE_MASK);
  x = atomKK->k_x.view<DeviceType>();
  f = atomKK->k_f.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();
  k_cutsq.template sync<DeviceType>();

  NeighListKokkos<DeviceType> *k_list = static_cast<NeighListKokkos<DeviceType> *>(list);
  d_numneigh = k_list->d_numneigh;
  d_neighbors = k_list->d_neighbors;
  d_ilist = k_list->d_ilist;
  inum = list->inum;

  need_dup = lmp->kokkos->need_dup<DeviceType>();
  // clang-format off
  if (need_dup) {
    dup_f     = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterDuplicated>(f);
    dup_vatom = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterDuplicated>(d_vatom);
  } else {
    ndup_f     = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterNonDuplicated>(f);
    ndup_vatom = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterNonDuplicated>(d_vatom);
    // clang-format on
  }

  //Precalc the max neighs. This is need to resize the jacobian.
  max_neighs = 0;
  Kokkos::parallel_reduce("PairSNAPKokkos::find_max_neighs", inum,
                          FindMaxNumNeighs<DeviceType>(k_list), Kokkos::Max<int>(max_neighs));

  // Handling batching
  chunk_size =
      MIN(chunksize, inum);    // chunksize is the maximum atoms per pass as defined by the user
  chunk_offset = 0;

  // Resize the jacobian, maybe we can get rid of this resize in every compute call. Do not initalize, we do so in the loop.
  Kokkos::realloc(Kokkos::WithoutInitializing, d_moment_jacobian, chunk_size,
                  alpha_index_basic_count, max_neighs, 3);

  // ========== Begin Core Computation ==========
  while (chunk_offset < inum) {    // batching to prevent OOM on device
    if (chunk_size > inum - chunk_offset) chunk_size = inum - chunk_offset;

    // First reset the working arrays from the last batch
    Kokkos::Experimental::fill(Kokkos::DefaultExecutionSpace(), d_moment_jacobian, 0.0);
    Kokkos::Experimental::fill(Kokkos::DefaultExecutionSpace(), d_moment_tensor_vals, 0.0);
    Kokkos::Experimental::fill(Kokkos::DefaultExecutionSpace(), d_nbh_energy_ders_wrt_moments, 0.0);

    // ========== Calculate the alpha basis (Per outer-atom parallelizaton) ==========

    chunk_offset += chunk_size;    // Manage halt condition
  }    // end batching while loop
  // ========== End Core Computation ==========

  //Clean up scatter views
}

// ========== Kernels ==========

// Finds the maximum number of neighbours in all neigbhourhoods. This enables use to set the size (2nd index) of the jacobian.
template <class DeviceType> struct FindMaxNumNeighs {
  typedef DeviceType device_type;
  NeighListKokkos<DeviceType> k_list;

  FindMaxNumNeighs(NeighListKokkos<DeviceType> *nl) : k_list(*nl) {}
  ~FindMaxNumNeighs() { k_list.copymode = 1; }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int &ii, int &max_neighs) const
  {
    const int i = k_list.d_ilist[ii];
    const int num_neighs = k_list.d_numneigh[i];
    if (max_neighs < num_neighs) max_neighs = num_neighs;
  }
};