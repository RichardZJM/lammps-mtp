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

#include "Kokkos_StdAlgorithms.hpp"

using namespace LAMMPS_NS;

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
      error->all(FLERR, "Pair style mtp/kk can currently only run on a single CPU thread.");

    PairMTP::init_style();
    return;
  }

  if (force->newton_pair == 0) error->all(FLERR, "Pair style MTP requires newton pair on.");

  // neighbor list request for KOKKOS
  neighflag = lmp->kokkos->neighflag;

  auto request = neighbor->add_request(this, NeighConst::REQ_FULL);
  request->set_kokkos_host(std::is_same_v<DeviceType, LMPHostType> &&
                           !std::is_same_v<DeviceType, LMPDeviceType>);
  request->set_kokkos_device(std::is_same_v<DeviceType, LMPDeviceType>);
  if (neighflag == FULL) error->all(FLERR, "Must use half neighbor list style with pair mtp/kk.");
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

template <class DeviceType> double PairMTPKokkos<DeviceType>::init_one(int i, int j)
{
  double cutone = PairMTP::init_one(i, j);
  //Don't need to do anything with the cutoff because the MTP (and original MLIP package) only uses one cutoff for all species combos.
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
  // We may need to process in chunks to deal with memory limitations
  // For now we expect the user to specify the chunk size

  if (narg != 3 || LAMMPS_NS::utils::lowercase(arg[1]) != "chunksize")
    error->all(FLERR,
               "Pair mtp/kk requires 3 arguments {{potential_file} \"chunksize\" {chunksize}}.");

  chunk_size = utils::inumeric(FLERR, arg[2], true, lmp);

  PairMTP ::settings(
      1, arg);    // This also calls read_file which parses and loads the necessary arrays in host

  // ---------- Now we move arrays to device ----------
  // First we set up the index lists
  MemKK::realloc_kokkos(d_alpha_index_basic, "mtp/kk:alpha_index_basic", alpha_index_basic_count,
                        4);
  MemKK::realloc_kokkos(d_alpha_index_times, "mtp/kk:alpha_index_times", alpha_index_times_count,
                        4);
  MemKK::realloc_kokkos(d_alpha_moment_mapping, "mtp/kk:moment_mapping", alpha_scalar_count);

  // Setup the learned coefficients
  int radial_coeff_count = species_count * species_count * radial_basis_size * radial_func_count;
  MemKK::realloc_kokkos(d_radial_basis_coeffs, "mtp/kk:radial_coeffs", radial_coeff_count);
  MemKK::realloc_kokkos(d_species_coeffs, "mtp/kk:species_coeffs", species_count);
  MemKK::realloc_kokkos(d_linear_coeffs, "mtp/kk:linear_coeffs", alpha_scalar_count);

  //Setup the working arrays. It might be preferable for these to be scatter view
  MemKK::realloc_kokkos(d_moment_jacobian, "mtp/kk:moment_jacobian", chunk_size, 2,
                        alpha_index_basic_count,
                        3);    // Arbitrary initial value (to be reallocated with max neighs)
  MemKK::realloc_kokkos(d_moment_tensor_vals, "mtp/kk:moment_tensor_vals", chunk_size,
                        alpha_moment_count);
  MemKK::realloc_kokkos(d_nbh_energy_ders_wrt_moments, "mtp/kk:nbh_energy_ders_wrt_moments",
                        chunk_size, alpha_moment_count);

  //Declare host arrays
  auto h_alpha_index_basic = Kokkos::create_mirror_view(d_alpha_index_basic);
  auto h_alpha_index_times = Kokkos::create_mirror_view(d_alpha_index_times);
  auto h_alpha_moment_mapping = Kokkos::create_mirror_view(d_alpha_moment_mapping);
  auto h_radial_basis_coeffs = Kokkos::create_mirror_view(d_radial_basis_coeffs);
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
  Kokkos::deep_copy(d_radial_basis_coeffs, h_radial_basis_coeffs);
  Kokkos::deep_copy(d_species_coeffs, h_species_coeffs);
  Kokkos::deep_copy(d_linear_coeffs, h_linear_coeffs);
  // No need to deep copy the working buffers.
}

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
    const int num_neighs = k_list.d_numneigh(i);
    if (max_neighs < num_neighs) max_neighs = num_neighs;
  }
};

/* ----------------------------------------------------------------------
   This version is a straightforward implementation
   ---------------------------------------------------------------------- */

template <class DeviceType> void PairMTPKokkos<DeviceType>::compute(int eflag_in, int vflag_in)
{
  // If we are running on host we just use the base implementation
  if (host_flag) {
    atomKK->sync(Host, X_MASK | F_MASK | TYPE_MASK);
    PairMTP::compute(eflag_in, vflag_in);
    atomKK->modified(Host, F_MASK);
    return;
  }

  eflag = eflag_in;
  vflag = vflag_in;

  if (neighflag == FULL) no_virial_fdotr_compute = 1;

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
  if (newton_pair == false) error->all(FLERR, "PairMTPKokkos requires 'newton on'.");

  // Now, ensure the atom data is synced
  atomKK->sync(execution_space, X_MASK | F_MASK | TYPE_MASK);
  x = atomKK->k_x.view<DeviceType>();
  f = atomKK->k_f.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();

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

  //Precalc the max neighs. This is needed to resize the jacobian.
  max_neighs = 0;
  Kokkos::parallel_reduce("PairMTPKokkos::find_max_neighs", inum,
                          FindMaxNumNeighs<DeviceType>(k_list), Kokkos::Max<int>(max_neighs));

  // Handling batching
  chunk_size =
      MIN(chunk_size, inum);    // chunksize is the maximum atoms per pass as defined by the user
  chunk_offset = 0;

  // Team sizes. We specify 32 for 1 warp per thread block.
  // Maybe need 64 for AMD?
  int team_size_default = 1;
  int vector_length_default = 1;
  if (!host_flag) team_size_default = 64;

  // Resize the jacobian if the max_neighs isn't large enough. Do not initalize, we do so in the loop.
  if ((int) d_moment_jacobian.extent(1) < max_neighs)
    Kokkos::realloc(Kokkos::WithoutInitializing, d_moment_jacobian, chunk_size, max_neighs,
                    alpha_index_basic_count, 3);

  EV_FLOAT ev;

  // ========== Begin Main Computation ==========
  while (chunk_offset < inum) {    // batching to prevent OOM on device
    EV_FLOAT ev_tmp;
    if (chunk_size > inum - chunk_offset) chunk_size = inum - chunk_offset;

    // ========== Init working views as 0  ==========
    {

      typename Kokkos::MDRangePolicy<Kokkos::Rank<2>, DeviceType, TagPairMTPInitMomentValsDers>
          policy_moment_init({0, 0}, {chunk_size, alpha_moment_count});
      Kokkos::parallel_for("InitMomentValDers", policy_moment_init, *this);
    }

    // ========== Calculate the basic alphas (Per outer-atom parallelizaton) ==========
    {
      int team_size = team_size_default;
      int vector_length = vector_length_default;
      check_team_size_for<TagPairMTPComputeAlphaBasic>(chunk_size, team_size, vector_length);
      int radial_scratch_count = radial_basis_size * 2;    // Vals and derivative
      int dist_coords_scratch_count = 4 * max_alpha_index_basic;
      int scratch_size = scratch_size_helper<F_FLOAT>(
          team_size * (radial_scratch_count + dist_coords_scratch_count));
      Kokkos::TeamPolicy<DeviceType, TagPairMTPComputeAlphaBasic> policy_basic_alpha(chunk_size,
                                                                                     team_size);
      policy_basic_alpha = policy_basic_alpha.set_scratch_size(0, Kokkos::PerTeam(scratch_size));
      Kokkos::parallel_for("ComputeAlphaBasic", policy_basic_alpha, *this);
    }

    // ========== Calculate the non-elementary alphas (Per neighbourhood parallelizaton ) ==========
    // This can be parallelized with dependence analysis. Worth exploring later although it shouldn't make a big difference for atom count much lower than chunk_size.
    // We can also maybe put the moment tensor vals here in shared memory with a heiarchal parallel approach here. Might be too large.
    // We would also need to put nbh energy calculation in here.
    // Needs testing but might not be very effective for MTP levels less than 10. Might be expensive to copy shared memory.
    {
      typename Kokkos::RangePolicy<DeviceType, TagPairMTPComputeAlphaTimes> policy_times(
          0, chunk_size);
      Kokkos::parallel_for("ComputeAlphaTimes", policy_times, *this);
    }

    // ========== Set the scalar nbh ders wrt moments ==========
    {
      typename Kokkos::MDRangePolicy<Kokkos::Rank<2>, DeviceType, TagPairMTPSetScalarNbhDers>
          policy_nbh_init({0, 0}, {chunk_size, alpha_scalar_count});
      Kokkos::parallel_for("SetScalarNbhDers", policy_nbh_init, *this);
    }

    // ========== Calc the nbh ders wrt moments ==========
    {
      typename Kokkos::RangePolicy<DeviceType, TagPairMTPComputeNbhDers> policy_nbh_calc(
          0, chunk_size);
      Kokkos::parallel_for("ComputeNbhDers", policy_nbh_calc, *this);
    }

    // ========== Compute force (and convolve alphas to get energy if needed) ==========
    {
      if (evflag) {
        if (neighflag == HALF) {
          typename Kokkos::RangePolicy<DeviceType, TagPairMTPComputeForce<HALF, 1>> policy_force(
              0, chunk_size);
          Kokkos::parallel_reduce(policy_force, *this, ev_tmp);
        } else if (neighflag == HALFTHREAD) {
          typename Kokkos::RangePolicy<DeviceType, TagPairMTPComputeForce<HALFTHREAD, 1>>
              policy_force(0, chunk_size);
          Kokkos::parallel_reduce(policy_force, *this, ev_tmp);
        }
      } else {
        if (neighflag == HALF) {
          typename Kokkos::RangePolicy<DeviceType, TagPairMTPComputeForce<HALF, 0>> policy_force(
              0, chunk_size);
          Kokkos::parallel_for(policy_force, *this);
        } else if (neighflag == HALFTHREAD) {
          typename Kokkos::RangePolicy<DeviceType, TagPairMTPComputeForce<HALFTHREAD, 0>>
              policy_force(0, chunk_size);
          Kokkos::parallel_for(policy_force, *this);
        }
      }
    }
    ev += ev_tmp;
    chunk_offset += chunk_size;    // Manage halt condition
  }    // end batching while loop

  // ========== End Main Computation ==========

  if (need_dup) Kokkos::Experimental::contribute(f, dup_f);

  if (eflag_global) eng_vdwl += ev.evdwl;
  if (vflag_global) {
    virial[0] += ev.v[0];
    virial[1] += ev.v[1];
    virial[2] += ev.v[2];
    virial[3] += ev.v[3];
    virial[4] += ev.v[4];
    virial[5] += ev.v[5];
  }

  if (vflag_fdotr) pair_virial_fdotr_compute(this);

  if (eflag_atom) {
    k_eatom.template modify<DeviceType>();
    k_eatom.template sync<LMPHostType>();
  }

  if (vflag_atom) {
    if (need_dup) Kokkos::Experimental::contribute(d_vatom, dup_vatom);
    k_vatom.template modify<DeviceType>();
    k_vatom.template sync<LMPHostType>();
  }

  atomKK->modified(execution_space, F_MASK);

  copymode = 0;

  // free duplicated memory
  if (need_dup) {
    dup_f = decltype(dup_f)();
    dup_vatom = decltype(dup_vatom)();
  }
}

// ========== Kernels ==========

// Inits the working arrays: jacobian and moment vals to 0. (ders not needed.
template <class DeviceType>
KOKKOS_INLINE_FUNCTION void PairMTPKokkos<DeviceType>::operator()(TagPairMTPInitMomentValsDers,
                                                                  const int &ii, const int &k) const
{
  d_moment_tensor_vals(ii, k) = 0;
  d_nbh_energy_ders_wrt_moments(ii, k) = 0;
}

// Calculates the basic alphas
template <class DeviceType>
KOKKOS_INLINE_FUNCTION void PairMTPKokkos<DeviceType>::operator()(
    TagPairMTPComputeAlphaBasic,
    const typename Kokkos::TeamPolicy<DeviceType, TagPairMTPComputeAlphaBasic>::member_type &team)
    const
{
  // Extract the atom number
  int ii = team.league_rank();
  if (ii >= chunk_size) return;

  // Get information about the central atom
  const int i = d_ilist[ii + chunk_offset];
  const F_FLOAT xi[3] = {x(i, 0), x(i, 1), x(i, 2)};
  const int itype = type[i] - 1;    // switch to zero indexing
  const int jnum = d_numneigh(i);

  // If precomputing everything is too much memory, we can consider calculating dist powers and coord powers on-the-fly with pow.
  shared_double_2d s_radial_basis_vals(team.team_scratch(0), team.team_size(), radial_basis_size);
  shared_double_2d s_radial_basis_ders(team.team_scratch(0), team.team_size(), radial_basis_size);
  shared_double_2d s_dist_powers(team.team_scratch(0), team.team_size(), max_alpha_index_basic);
  shared_double_3d s_coord_powers(team.team_scratch(0), team.team_size(), max_alpha_index_basic);

  // Now we calculate the alpha basics. There might be benefits to using a parallel reduce into the array of moment values here.
  // However, in the case that there are more threads than alpha basics (MTP lvl 12 or more), we can offset the starting indices, and guarentee no contention without even needing atomics. Doing this also might help with memory coalescing?

  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, jnum), [=](const int jj) {
    const int j = d_neighbors(i, jj) & NEIGHMASK;
    const int jtype = type[j] - 1;    // switch to zero indexing
    const F_FLOAT r[3] = {x(j, 0) - xi[0], x(j, 1) - xi[1], x(j, 2) - xi[2]};
    const F_FLOAT rsq = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
    if (rsq > max_cutoff_sq) return;

    const F_FLOAT dist = sqrt(rsq);

    s_dist_powers(jj, 0) = s_coord_powers(jj, 0, 0) = s_coord_powers(jj, 0, 1) =
        s_coord_powers(jj, 0, 2) = 1;    // Set the constants

    // Precompute the coord and distance power
    for (int k = 1; k < max_alpha_index_basic; k++) {
      s_dist_powers(jj, k) = s_dist_powers(jj, k - 1) * dist;
      for (int a = 0; a < 3; a++) s_coord_powers(jj, k, a) = s_coord_powers(jj, k - 1, a) * r[a];
    }

    // ---------- Calculate the radial basis functions ----------
    // Currently, I just have it hard coded for Rb_Chebyshev. I'll need to implement a way to handle different radial basis sets in kokkos

    // Calculate the radial basis and store in shared memory
    F_FLOAT mult = 2.0 / (max_cutoff - min_cutoff);
    F_FLOAT ksi = (2 * dist - (min_cutoff + max_cutoff)) / (max_cutoff - min_cutoff);

    s_radial_basis_vals(jj, 0) = scaling * (1 * (dist - max_cutoff) * (dist - max_cutoff));
    s_radial_basis_vals(jj, 1) = scaling * (ksi * (dist - max_cutoff) * (dist - max_cutoff));
    for (int k = 2; k < radial_basis_size; k++) {
      s_radial_basis_vals(jj, k) =
          2 * ksi * s_radial_basis_vals(jj, k - 1) - s_radial_basis_vals(jj, k - 2);
    }

    // Do the same with the derivatives
    s_radial_basis_ders(jj, 0) =
        scaling * (0 * (dist - max_cutoff) * (dist - max_cutoff) + 2 * (dist - max_cutoff));
    s_radial_basis_ders(jj, 1) = scaling *
        (mult * (dist - max_cutoff) * (dist - max_cutoff) + 2 * ksi * (dist - max_cutoff));
    for (int k = 2; k < radial_basis_size; k++) {
      s_radial_basis_ders(jj, k) =
          2 * (mult * s_radial_basis_vals(jj, k - 1) + ksi * s_radial_basis_ders(jj, k - 1)) -
          s_radial_basis_ders(jj, k - 2);
    }

    //Now, we loop through all the basic alphas
    for (int k = 0; k < alpha_index_basic_count; k++) {

      F_FLOAT val = 0;
      F_FLOAT der = 0;
      int mu = d_alpha_index_basic(k, 0);
      int a0 = d_alpha_index_basic(k, 1);
      int a1 = d_alpha_index_basic(k, 2);
      int a2 = d_alpha_index_basic(k, 3);

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
      F_FLOAT norm_fac = 1.0 / s_dist_powers(jj, norm_rank);
      val *= norm_fac;
      der = der * norm_fac - norm_rank * val / dist;

      F_FLOAT pow0 = s_coord_powers(jj, a0, 0);
      F_FLOAT pow1 = s_coord_powers(jj, a1, 1);
      F_FLOAT pow2 = s_coord_powers(jj, a2, 2);
      F_FLOAT pow = pow0 * pow1 * pow2;
      Kokkos::atomic_add(&d_moment_tensor_vals(ii, k), val * pow);

      // Get the component's derivatives too
      F_FLOAT temp_jac[3] = {pow * r[0], pow * r[1], pow * r[2]};

      pow *= der / dist;
      temp_jac[0] = pow * r[0];
      temp_jac[1] = pow * r[1];
      temp_jac[2] = pow * r[2];

      if (a0 != 0) temp_jac[0] += val * a0 * s_coord_powers(jj, a0 - 1, 0) * pow1 * pow2;
      if (a1 != 0) temp_jac[1] += val * a1 * pow0 * s_coord_powers(jj, a1 - 1, 1) * pow2;
      if (a2 != 0) temp_jac[2] += val * a2 * pow0 * pow1 * s_coord_powers(jj, a2 - 1, 2);

      d_moment_jacobian(ii, jj, k, 0) = temp_jac[0];
      d_moment_jacobian(ii, jj, k, 1) = temp_jac[1];
      d_moment_jacobian(ii, jj, k, 2) = temp_jac[2];

      // This version uses 2 less registers but runs slightly slower
      // pow *= der / dist;

      // F_FLOAT temp_jac = pow * r[0];
      // if (a0 != 0) temp_jac += val * a0 * s_coord_powers(jj, a0 - 1, 0) * pow1 * pow2;
      // d_moment_jacobian(ii, jj, k, 0) = temp_jac;

      // temp_jac = pow * r[1];
      // if (a1 != 0) temp_jac += val * a1 * pow0 * s_coord_powers(jj, a1 - 1, 1) * pow2;
      // d_moment_jacobian(ii, jj, k, 1) = temp_jac;

      // temp_jac = pow * r[2];
      // if (a2 != 0) temp_jac += val * a2 * pow0 * pow1 * s_coord_powers(jj, a2 - 1, 2);
      // d_moment_jacobian(ii, jj, k, 2) = temp_jac;
    }
  });
}

// Calculates the non-elementary alpha from the basic alphas
template <class DeviceType>
KOKKOS_INLINE_FUNCTION void PairMTPKokkos<DeviceType>::operator()(TagPairMTPComputeAlphaTimes,
                                                                  const int &ii) const
{
  // Traverse all edges in the alpha times compute graph
  for (int k = 0; k < alpha_index_times_count; k++) {
    int a0 = d_alpha_index_times(k, 0);
    int a1 = d_alpha_index_times(k, 1);
    int multipiler = d_alpha_index_times(k, 2);
    int a3 = d_alpha_index_times(k, 3);

    F_FLOAT val0 = d_moment_tensor_vals(ii, a0);
    F_FLOAT val1 = d_moment_tensor_vals(ii, a1);

    d_moment_tensor_vals(ii, a3) += multipiler * val0 * val1;
  }
}

// Sets the nbh energy ders as the linear coeffs
template <class DeviceType>
KOKKOS_INLINE_FUNCTION void PairMTPKokkos<DeviceType>::operator()(TagPairMTPSetScalarNbhDers,
                                                                  const int &ii, const int &k) const
{
  d_nbh_energy_ders_wrt_moments(ii, d_alpha_moment_mapping(k)) = d_linear_coeffs(k);
}

template <class DeviceType>
KOKKOS_INLINE_FUNCTION void PairMTPKokkos<DeviceType>::operator()(TagPairMTPComputeNbhDers,
                                                                  const int &ii) const
{
  for (int k = alpha_index_times_count - 1; k >= 0; k--) {
    int a0 = d_alpha_index_times(k, 0);
    int a1 = d_alpha_index_times(k, 1);
    int multipiler = d_alpha_index_times(k, 2);
    int a3 = d_alpha_index_times(k, 3);

    F_FLOAT val0 = d_moment_tensor_vals(ii, a0);
    F_FLOAT val1 = d_moment_tensor_vals(ii, a1);
    F_FLOAT val3 = d_nbh_energy_ders_wrt_moments(ii, a3);

    d_nbh_energy_ders_wrt_moments(ii, a1) += val3 * multipiler * val0;
    d_nbh_energy_ders_wrt_moments(ii, a0) += val3 * multipiler * val1;
  }
}

template <class DeviceType>
template <int NEIGHFLAG, int EVFLAG>
KOKKOS_INLINE_FUNCTION void
PairMTPKokkos<DeviceType>::operator()(TagPairMTPComputeForce<NEIGHFLAG, EVFLAG>, const int &ii,
                                      EV_FLOAT &ev) const
{

  // The f array is duplicated for OpenMP, atomic for GPU, and neither for Serial
  auto v_f =
      ScatterViewHelper<NeedDup_v<NEIGHFLAG, DeviceType>, decltype(dup_f), decltype(ndup_f)>::get(
          dup_f, ndup_f);
  auto a_f = v_f.template access<AtomicDup_v<NEIGHFLAG, DeviceType>>();

  const int i = d_ilist[ii + chunk_offset];
  const int itype = type(i) - 1;    // zero indexing
  const F_FLOAT xi[3] = {x(i, 0), x(i, 1), x(i, 2)};
  const int jnum = d_numneigh(i);

  for (int jj = 0; jj < jnum; jj++) {
    const int j = d_neighbors(i, jj) & NEIGHMASK;

    F_FLOAT r[3] = {x(j, 0) - xi[0], x(j, 1) - xi[1], x(j, 2) - xi[2]};
    const F_FLOAT rsq = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
    if (rsq > max_cutoff_sq) continue;

    F_FLOAT temp_force[3] = {0, 0, 0};
    for (int k = 0; k < alpha_index_basic_count; k++) {
      for (int a = 0; a < 3; a++) {
        //Calculate forces
        temp_force[a] += d_nbh_energy_ders_wrt_moments(ii, k) * d_moment_jacobian(ii, jj, k, a);
      }
    }

    a_f(i, 0) += temp_force[0];
    a_f(i, 1) += temp_force[1];
    a_f(i, 2) += temp_force[2];

    a_f(j, 0) -= temp_force[0];
    a_f(j, 1) -= temp_force[1];
    a_f(j, 2) -= temp_force[2];

    if (EVFLAG && eflag_either) {
      v_tally_xyz<NEIGHFLAG>(ev, i, j, temp_force[0], temp_force[1], temp_force[2], r[0], r[1],
                             r[2]);
    }
  }

  if (EVFLAG && eflag_either) {

    F_FLOAT nbh_energy =
        d_species_coeffs[itype];    // Essentially the reference point energy per species

    // Take the linear combination of the basis set and the learned coefficients. This might benefit from using a reduction?
    for (int k = 0; k < alpha_scalar_count; k++) {
      int basis_member_index = d_alpha_moment_mapping(k);
      nbh_energy += d_linear_coeffs(k) * d_moment_tensor_vals(ii, basis_member_index);
    }

    if (eflag_global) ev.evdwl += nbh_energy;
    if (eflag_atom) d_eatom[i] += nbh_energy;
  }
}

template <class DeviceType>
template <int NEIGHFLAG, int EVFLAG>
KOKKOS_INLINE_FUNCTION void
PairMTPKokkos<DeviceType>::operator()(TagPairMTPComputeForce<NEIGHFLAG, EVFLAG>,
                                      const int &ii) const
{
  EV_FLOAT ev;
  this->template operator()<NEIGHFLAG, EVFLAG>(TagPairMTPComputeForce<NEIGHFLAG, EVFLAG>(), ii, ev);
}

// =========== Helper Functions (Also used in other Kokkos potentials)===========
template <class DeviceType>
template <int NEIGHFLAG>
KOKKOS_INLINE_FUNCTION void
PairMTPKokkos<DeviceType>::v_tally_xyz(EV_FLOAT &ev, const int &i, const int &j, const F_FLOAT &fx,
                                       const F_FLOAT &fy, const F_FLOAT &fz, const F_FLOAT &delx,
                                       const F_FLOAT &dely, const F_FLOAT &delz) const
{
  // The vatom array is duplicated for OpenMP, atomic for GPU, and neither for Serial

  auto v_vatom = ScatterViewHelper<NeedDup_v<NEIGHFLAG, DeviceType>, decltype(dup_vatom),
                                   decltype(ndup_vatom)>::get(dup_vatom, ndup_vatom);
  auto a_vatom = v_vatom.template access<AtomicDup_v<NEIGHFLAG, DeviceType>>();

  const E_FLOAT v0 = delx * fx;
  const E_FLOAT v1 = dely * fy;
  const E_FLOAT v2 = delz * fz;
  const E_FLOAT v3 = delx * fy;
  const E_FLOAT v4 = delx * fz;
  const E_FLOAT v5 = dely * fz;

  if (vflag_global) {
    ev.v[0] += v0;
    ev.v[1] += v1;
    ev.v[2] += v2;
    ev.v[3] += v3;
    ev.v[4] += v4;
    ev.v[5] += v5;
  }

  if (vflag_atom) {
    a_vatom(i, 0) += 0.5 * v0;
    a_vatom(i, 1) += 0.5 * v1;
    a_vatom(i, 2) += 0.5 * v2;
    a_vatom(i, 3) += 0.5 * v3;
    a_vatom(i, 4) += 0.5 * v4;
    a_vatom(i, 5) += 0.5 * v5;
    a_vatom(j, 0) += 0.5 * v0;
    a_vatom(j, 1) += 0.5 * v1;
    a_vatom(j, 2) += 0.5 * v2;
    a_vatom(j, 3) += 0.5 * v3;
    a_vatom(j, 4) += 0.5 * v4;
    a_vatom(j, 5) += 0.5 * v5;
  }
}

template <class DeviceType>
template <class TagStyle>
void PairMTPKokkos<DeviceType>::check_team_size_for(int inum, int &team_size, int vector_length)
{
  int team_size_max;

  team_size_max = Kokkos::TeamPolicy<DeviceType, TagStyle>(inum, Kokkos::AUTO)
                      .team_size_max(*this, Kokkos::ParallelForTag());

  if (team_size * vector_length > team_size_max) team_size = team_size_max / vector_length;
}

template <class DeviceType>
template <typename scratch_type>
int PairMTPKokkos<DeviceType>::scratch_size_helper(int values_per_team)
{
  typedef Kokkos::View<scratch_type *, Kokkos::DefaultExecutionSpace::scratch_memory_space,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      ScratchViewType;

  return ScratchViewType::shmem_size(values_per_team);
}    // namespace LAMMPS_NS

/* ---------------------------------------------------------------------- */

namespace LAMMPS_NS {
template class PairMTPKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PairMTPKokkos<LMPHostType>;
#endif
}    // namespace LAMMPS_NS