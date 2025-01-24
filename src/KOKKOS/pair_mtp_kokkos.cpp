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

    // ========== Calculate the basis alphas (Per outer-atom parallelizaton) ==========

    chunk_offset += chunk_size;    // Manage halt condition
  }    // end batching while loop
  // ========== End Core Computation ==========

  //Clean up scatter views
}

// ========== Kernels ==========

// Finds the maximum number of neighbours in all neigbhourhoods. This enables use to set the size (2nd index) of the jacobian. (Copied from other potentials)
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

// Calculates the basic alphas
template <class DeviceType>
KOKKOS_INLINE_FUNCTION void PairMTPKokkos<DeviceType>::operator()(
    CalcAlphaBasic,
    const typename Kokkos::TeamPolicy<DeviceType, CalcAlphaBasic>::member_type &team) const
{
  // Extract the atom number
  int ii = team.team_rank() + team.league_rank() * team.team_size();
  if (ii >= chunk_size) return;

  // Get information about the central atom
  const int i = d_ilist[ii + chunk_offset] - 1;    // switch to zero indexing
  const double xi[3] = {x[i][0], x[i][1], x[i][2]};
  const int itype = type[i];
  const int num_neighs = d_numneigh[i];

  // If precomputing everything is too much memory, we can consider calculating dist powers and coord powers on-the-fly?
  shared_double_2d s_radial_basis_vals(team.team_scratch(0), team.team_size(), radial_basis_size);
  shared_double_2d s_radial_basis_ders(team.team_scratch(0), team.team_size(), radial_basis_size);
  shared_double_2d s_dist_powers(team.team_scratch(0), team.team_size(), max_alpha_index_basic);
  shared_double_3d s_coord_powers(team.team_scratch(0), team.team_size(), max_alpha_index_basic, 3);

  // Now we calculate the alpha basics. There might be benefits to using a parallel reduce into the array of moment values here.
  // However, in the case that there are more threads than alpha basics (MTP lvl 12 or more), we can offset the starting indices, and guarentee no contention without even needing atomics. Doing this also might help with memory coalescing?

  Kokkos::parallel_for(TeamThreadTrange(team, jnum), [=](const int jj) {
    int j = d_neighbors(i, jj);
    j &= NEIGHMASK;
    const int jtype = type(j) - 1;    // switch to zero indexing
    const double r[3] = {x(j, 0) - xi[0], x(j, 1) - xi[1], x(j, 2) - xi[2]};
    const F_FLOAT rsq = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
    if (rsq < cutsq(i + 1, j + 1)) return;

    const F_FLOAT dist = sqrt(rsq);

    s_dist_powers(jj, 0) = s_coord_powers(jj, 0) = s_coord_powers(jj, 1) = s_coord_powers(jj, 2) =
        1;    // Set the constants

    // Precompute the coord and distance power
    int thread_offset = k * s_radial_basis_vals;
    for (int k = 1; k < max_alpha_index_basic; k++) {
      s_dist_powers(jj, k) = s_dist_powers(jj, k - 1) * dist;
      for (int a = 0; a < 3; a++) s_coord_powers(jj, k, a) = s_coord_powers(jj, k - 1, a) * r[a];
    }

    // ---------- Calculate the radial basis functions ----------
    // Currently, I just have it hard coded for Rb_Chebyshev. I need to implement a way to handle different radial basis sets in kokkos

    // Calculate the radial basis and store in shared memory
    F_FLOAT mult = 2.0 / (max_cutoff - min_cutoff);
    F_FLOAT ksi = (2 * dist - (min_cutoff + max_cutoff)) / (max_cutoff - min_cutoff);

    s_radial_basis_vals(jj, 0) = scaling * (1 * (dist - max_cutoff) * (dist - max_cutoff));
    s_radial_basis_vals(jj, 1) = scaling * (ksi * (dist - max_cutoff) * (dist - max_cutoff));
    for (int k = 2; k < size; k++) {
      s_radial_basis_vals(jj, k) =
          2 * ksi * s_radial_basis_vals(jj, k - 1) - s_radial_basis_vals(jj, k - 2);
    }

    // Do the same with the derivatives
    s_radial_basis_ders(jj, 0) =
        scaling * (0 * (dist - max_cutoff) * (dist - max_cutoff) + 2 * (dist - max_cutoff));
    s_radial_basis_ders(jj, 1) = scaling *
        (mult * (dist - max_cutoff) * (dist - max_cutoff) + 2 * ksi * (dist - max_cutoff));
    for (int k = 2; k < size; k++) {
      s_radial_basis_ders(jj, k) =
          2 * (mult * s_radial_basis_vals(jj, k - 1) + ksi * s_radial_basis_ders(jj, k - 1)) -
          s_radial_basis_ders(jj, k - 2);
    }

    //Now, we loop through all the basic alphas
    // To reduce contention we are going to offset the starting index
    int startIndex = jj % team_size;
    for (int kk = startIndex; kk < startIndex + alpha_index_basic_count; kk++) {
      int k = int index = k % alpha_index_basic_count;
      F_FLOAT val = 0;
      F_FLOAT der = 0;
      int mu = alpha_index_basic(k, 0);
      int a0 = alpha_index_basic(k, 1);
      int a1 = alpha_index_basic(k, 2);
      int a2 = alpha_index_basic(k, 3);

      //Find the offset for the radial basis coeffs
      int pair_offset = itype * species_count + jtype;
      int offset = (pair_offset * radial_basis_size * radial_func_count) + mu * radial_basis_size;

      // Find the radial component and its derivative
      for (int ri = 0; ri < radial_basis_size; ri++) {
        val += d_radial_basis_coeffs(offset + ri) * s_radial_basis_vals(jj, ri);
        der += d_radial_basis_coeffs(offset + ri) * s_radial_basis_ders(jj, ri);
      }

      // Normalize by the rank of alpha's coresponding tensor
      int norm_rank = a0 + a1 + a2;
      F_FLOAT norm_fac = 1.0 / s_dist_powers(jj, ai);
      val *= norm_fac;
      der = der * norm_fac - norm_rank * val / dist;

      F_FLOAT pow0 = s_coord_powers(a0, 0);
      F_FLOAT pow1 = s_coord_powers(a1, 1);
      F_FLOAT pow2 = s_coord_powers(a2, 2);
      F_FLOAT pow = pow0 * pow1 * pow2;
      d_moment_tensor_vals(ii, k) += val * pow;

      // Get the component's derivatives too
      pow *= der / dist;
      d_moment_jacobian(ii, k, jj, 0) += pow * r[0];
      d_moment_jacobian(ii, k, jj, 1) += pow * r[1];
      d_moment_jacobian(ii, k, jj, 2) += pow * r[2];

      // I've removed branch divergence here but maybe an if statement approach is better here, depending on contention.
      Kokkos::atomic_add(&d_moment_jacobian(ii, k, jj, 0),
                         val * a0 * s_coord_powers(jj, Kokkos::max(a0 - 1, 1), 0) * pow1 * pow2);
      Kokkos::atomic_add(&d_moment_jacobian(ii, k, jj, 1),
                         val * a1 * pow0 * s_coord_powers(jj, Kokkos::max(a1 - 1, 1), 1) * pow2);
      Kokkos::atomic_add(&d_moment_jacobian(ii, k, jj, 2),
                         val * a2 * pow0 * pow1 * s_coord_powers(jj, Kokkos::max(a2 - 1, 1), 2));

      // TODO: Try using the following too if applicable where
      // There is no contention for team size of 32 and MTP lvl >= 12,
      // d_moment_jacobian(ii, k, jj, 0) +=
      //     val * a0 * s_coord_powers(jj, Kokkos::max(a0 - 1, 1), 0) * pow1 * pow2;
      // d_moment_jacobian(ii, k, jj, 1) +=
      //     val * a1 * pow0 * s_coord_powers(jj, Kokkos::max(a1 - 1, 1), 1) * pow2;
      // d_moment_jacobian(ii, k, jj, 2) +=
      //     val * a2 * pow0 * pow1 * s_coord_powers(jj, Kokkos::max(a2 - 1, 1), 2);
    }
  });
};
