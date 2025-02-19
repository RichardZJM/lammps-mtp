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

#include "pair_mtp_extrapolation.h"

#include "mtp_radial_basis.h"
#include "mtp_rb_chebyshev_basis.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"

#include <cmath>
#include <csignal>
#include <fstream>
#include <iostream>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairMTPExtrapolation::~PairMTPExtrapolation()
{
  if (allocated) {
    memory->destroy(active_set);
    memory->destroy(inverse_active_set);
    memory->destroy(radial_jacobian);
    memory->destroy(energy_ders_wrt_coeffs);
    if (!pool_grades) memory->destroy(nbh_extrapolation_grades);
  }
}

/* ----------------------------------------------------------------------
   Straightfoward MTP implementation based on MLIP3
   ---------------------------------------------------------------------- */

void PairMTPExtrapolation::compute(int eflag, int vflag)
{
  // Simply call the base class compute if we aren't sampling this time step
  steps_since_last_sample++;
  if (steps_since_last_sample < sampling_frequency) {
    PairMTP::compute(eflag, vflag);
    return;
  }

  ev_setup(eflag, vflag);

  double **x = atom->x;      // atomic positons
  double **f = atom->f;      // atomic forces
  int *type = atom->type;    //atomic types

  // int nlocal = atom->nlocal; // Don't really need this
  // int newton_pair = force->newton_pair; // Newton pair is forced on

  int inum = list->inum;             // The number of central atoms (neigbhourhoods)
  int *ilist = list->ilist;          // List of the central atoms in order
  int *numneigh = list->numneigh;    // List of the number of neighbours for each central atom
  int **firstneigh =
      list->firstneigh;    //List  (head of array) of neighbours for a given central atom

  // Resize the nbh extrapolation grades if needed. No need to initialize, first access is write
  if (!pool_grades && nbh_count < inum) {
    memory->grow(nbh_extrapolation_grades, inum, "nbh_extrapolation_grades");
    nbh_count = inum;
  }

  // If are in configuration, we need to reset the working array once per compute call / config
  if (pool_grades)
    std::fill(&energy_ders_wrt_coeffs[0], &energy_ders_wrt_coeffs[0] + coeff_count, 0.0);

  // Loop over all provided neighbourhoods
  for (int ii = 0; ii < inum; ii++) {
    const int i = ilist[ii];          // Set central atom index
    const int itype = type[i] - 1;    // Set central atom type. Convert back to zero indexing.
    if (itype >= species_count)
      error->all(FLERR,
                 "Too few species count in the MTP potential!");    // Might not need this check
    int jnum = numneigh[i];                                         // Set number of neighbours
    double nbh_energy = 0;
    const double xi[3] = {x[i][0], x[i][1],
                          x[i][2]};    // Cache the position of the central atom for efficiency

    // Resize the jacobian and cutoff if needed. No need to initialize, first access is write
    if (jac_size < jnum) {
      memory->grow(moment_jacobian, alpha_index_basic_count, jnum, 3, "moment_jacobian");
      memory->grow(within_cutoff, jnum, "within_cutoff");
      jac_size = jnum;
    }

    // Reset the working arrays
    std::fill(&moment_tensor_vals[0], &moment_tensor_vals[0] + alpha_moment_count, 0.0);
    std::fill(&nbh_energy_ders_wrt_moments[0], &nbh_energy_ders_wrt_moments[0] + alpha_moment_count,
              0.0);
    std::fill(&radial_moment_ders[0], &radial_moment_ders[0] + alpha_moment_count, 0.0);
    std::fill(&radial_jacobian[0][0][0],
              &radial_jacobian[0][0][0] +
                  (alpha_index_basic_count * species_count * radial_coeff_count_per_pair),
              0.0);

    if (!pool_grades)
      std::fill(&energy_ders_wrt_coeffs[0], &energy_ders_wrt_coeffs[0] + coeff_count, 0.0);

    // ------------ Begin Alpha Basic Calc ------------
    // Loop over all neighbours
    for (int jj = 0; jj < jnum; jj++) {
      int j = firstneigh[i][jj];    //List of neighbours
      j &= NEIGHMASK;
      const int jtype = type[j] - 1;    // Convert back to zero indexing
      if (jtype >= species_count)
        error->all(FLERR,
                   "Too few species count in the MTP potential!");    // Might not need this check

      const double r[3] = {x[j][0] - xi[0], x[j][1] - xi[1], x[j][2] - xi[2]};
      const double rsq = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];

      if (rsq > cutsq[itype + 1][jtype + 1]) {    //1 indexing
        within_cutoff[jj] = false;
        continue;
      }
      within_cutoff[jj] = true;

      const double dist = std::sqrt(rsq);
      radial_basis->calc_radial_basis_ders(dist);    // Calculate radial basis

      // Precompute the coord and distance power
      for (int k = 1; k < max_alpha_index_basic; k++) {
        dist_powers[k] = dist_powers[k - 1] * dist;
        for (int a = 0; a < 3; a++) coord_powers[k][a] = coord_powers[k - 1][a] * r[a];
      }

      //Calculate the alpha basics
      for (int k = 0; k < alpha_index_basic_count; k++) {
        double val = 0;
        double der = 0;
        int mu = alpha_index_basic[k][0];

        // Normalize by the rank of alpha's coresponding tensor
        int norm_rank = alpha_index_basic[k][1] + alpha_index_basic[k][2] + alpha_index_basic[k][3];
        double norm_fac = 1.0 / dist_powers[norm_rank];

        double pow0 = coord_powers[alpha_index_basic[k][1]][0];
        double pow1 = coord_powers[alpha_index_basic[k][2]][1];
        double pow2 = coord_powers[alpha_index_basic[k][3]][2];
        double pow = pow0 * pow1 * pow2;

        //Find the offset for the radial basis coeffs
        int pair_offset = itype * species_count + jtype;
        int mu_offset = mu * radial_basis_size;
        int offset = (pair_offset * radial_basis_size * radial_func_count) + mu_offset;

        // Find the radial component and its derivative
        for (int ri = 0; ri < radial_basis_size; ri++) {
          val += radial_basis_coeffs[offset + ri] * radial_basis->radial_basis_vals[ri];
          der += radial_basis_coeffs[offset + ri] * radial_basis->radial_basis_ders[ri];
          radial_jacobian[k][jtype][mu_offset + ri] +=
              radial_basis->radial_basis_vals[ri] * norm_fac * pow;
        }

        val *= norm_fac;
        der = der * norm_fac - norm_rank * val / dist;
        moment_tensor_vals[k] += val * pow;

        // Get the component's derivatives too
        pow *= der / dist;
        moment_jacobian[k][jj][0] = pow * r[0];
        moment_jacobian[k][jj][1] = pow * r[1];
        moment_jacobian[k][jj][2] = pow * r[2];

        if (alpha_index_basic[k][1] != 0) {
          moment_jacobian[k][jj][0] += val * alpha_index_basic[k][1] *
              coord_powers[alpha_index_basic[k][1] - 1][0] * pow1 * pow2;
        }    //Chain rule for nonzero rank
        if (alpha_index_basic[k][2] != 0) {
          moment_jacobian[k][jj][1] += val * alpha_index_basic[k][2] * pow0 *
              coord_powers[alpha_index_basic[k][2] - 1][1] * pow2;
        }    //Chain rule for nonzero rank
        if (alpha_index_basic[k][3] != 0) {
          moment_jacobian[k][jj][2] += val * alpha_index_basic[k][3] * pow0 * pow1 *
              coord_powers[alpha_index_basic[k][3] - 1][2];
        }    //Chain rule for nonzero rank
      }
    }

    // ------------ Contruct Other Alphas  ------------
    for (int k = 0; k < alpha_index_times_count; k++) {
      double val0 = moment_tensor_vals[alpha_index_times[k][0]];
      double val1 = moment_tensor_vals[alpha_index_times[k][1]];
      int val2 = alpha_index_times[k][2];
      moment_tensor_vals[alpha_index_times[k][3]] += val2 * val0 * val1;
    }

    // ------------ Convolve Basis Set From Alpha Map ------------
    // Calculate the energies only if needed. We always need the basis member for extrapolation.
    int linear_basis_offset = radial_coeff_count + species_count;
    if (eflag_either) {
      for (int k = 0; k < alpha_scalar_count; k++) {
        double basis_member = moment_tensor_vals[alpha_moment_mapping[k]];
        energy_ders_wrt_coeffs[linear_basis_offset + k] += basis_member;
        nbh_energy += linear_coeffs[k] * basis_member;
      }
      // Tally energies per flags if needed
      if (eflag_atom) eatom[i] = nbh_energy;
      if (eflag_global) eng_vdwl += nbh_energy;
    } else
      for (int k = 0; k < alpha_scalar_count; k++)
        energy_ders_wrt_coeffs[linear_basis_offset + k] +=
            moment_tensor_vals[alpha_moment_mapping[k]];

    // ------------ Also add the species coefficient ------------
    energy_ders_wrt_coeffs[radial_coeff_count + itype] += 1;    //species_coeffs[itype];

    // =========== Begin Backpropogation ===========

    //------------ Step 1: NBH energy derivative is the corresponding linear combination------------
    for (int k = 0; k < alpha_scalar_count; k++)
      nbh_energy_ders_wrt_moments[alpha_moment_mapping[k]] = linear_coeffs[k];

    //------------ Step 2: Propogate chain rule through the alpha times to the alpha basics ------------
    for (int k = alpha_index_times_count - 1; k >= 0; k--) {
      int a0 = alpha_index_times[k][0];
      int a1 = alpha_index_times[k][1];
      int multipiler = alpha_index_times[k][2];
      int a3 = alpha_index_times[k][3];

      double val0 = moment_tensor_vals[a0];
      double val1 = moment_tensor_vals[a1];
      double val3 = nbh_energy_ders_wrt_moments[a3];

      nbh_energy_ders_wrt_moments[a1] += val3 * multipiler * val0;
      nbh_energy_ders_wrt_moments[a0] += val3 * multipiler * val1;
    }

    //------------ Step 3: Multiply energy ders wrt moment by the moment jacobian to get forces ------------
    for (int jj = 0; jj < jnum; jj++) {
      int j = firstneigh[i][jj];
      j &= NEIGHMASK;
      if (!within_cutoff[jj]) continue;

      double temp_force[3] = {0, 0, 0};
      for (int k = 0; k < alpha_index_basic_count; k++) {
        // Backprop relative to positions for forces
        for (int a = 0; a < 3; a++) {
          //Calculate forces
          temp_force[a] += nbh_energy_ders_wrt_moments[k] * moment_jacobian[k][jj][a];
        }
      }

      f[i][0] += temp_force[0];
      f[i][1] += temp_force[1];
      f[i][2] += temp_force[2];

      f[j][0] -= temp_force[0];
      f[j][1] -= temp_force[1];
      f[j][2] -= temp_force[2];

      //Calculate virial stress
      if (vflag) {
        // We only need to calculate rel pos again if stress are needed
        const double r[3] = {x[j][0] - xi[0], x[j][1] - xi[1], x[j][2] - xi[2]};
        virial[0] -= temp_force[0] * r[0] * 2;    //xx
        virial[1] -= temp_force[1] * r[1] * 2;    //yy
        virial[2] -= temp_force[2] * r[2] * 2;    //zz

        virial[3] -= (temp_force[0] * r[1] + temp_force[1] * r[0]);    //xy
        virial[4] -= (temp_force[0] * r[2] + temp_force[2] * r[0]);    //xz
        virial[5] -= (temp_force[1] * r[2] + temp_force[2] * r[1]);    //yz

        //This can be more efficient but I'm not sure if it's even needed.
        if (vflag_atom) {
          vatom[i][0] -= temp_force[0] * r[0] * 2;    //xx
          vatom[i][1] -= temp_force[1] * r[1] * 2;    //yy
          vatom[i][2] -= temp_force[2] * r[2] * 2;    //zz

          vatom[i][3] -= (temp_force[0] * r[1] + temp_force[1] * r[0]);    //xy
          vatom[i][4] -= (temp_force[0] * r[2] + temp_force[2] * r[0]);    //xz
          vatom[i][5] -= (temp_force[1] * r[2] + temp_force[2] * r[1]);    //yz
        }
      }
    }

    //------------ Step 3.5: Multiply energy ders wrt moment by the radial jacobian to get rad ders ------------
    for (int k = 0; k < alpha_index_basic_count; k++)
      for (int jjtype = 0; jjtype < species_count; jjtype++) {
        int offset = (itype * species_count + jjtype) * radial_coeff_count_per_pair;
        for (int ri = 0; ri < radial_coeff_count_per_pair; ri++)
          energy_ders_wrt_coeffs[offset + ri] +=
              nbh_energy_ders_wrt_moments[k] * radial_jacobian[k][jjtype][ri];
      }

    // Directly calculate extraplation grade for neighbourhood mode
    if (!pool_grades) {
      max_grade = 0;
      double grade = calculate_extrapolation_grade(energy_ders_wrt_coeffs);
      max_grade = std::max(grade, max_grade);
      nbh_extrapolation_grades[ii] = grade;
    }
  }

  // MPI reduce operations based on selection mode
  if (pool_grades) {    // Configuration mode
    MPI_Allreduce(MPI_IN_PLACE, &energy_ders_wrt_coeffs[0], coeff_count, MPI_DOUBLE, MPI_SUM,
                  world);
    if (comm->me == 0) max_grade = calculate_extrapolation_grade(energy_ders_wrt_coeffs);
  } else {    // Neighbourhood mode
    MPI_Allreduce(MPI_IN_PLACE, &max_grade, 1, MPI_DOUBLE, MPI_MAX, world);
  }

  if (comm->me == 0) {
    if (max_grade >= select_threshold)
      utils::logmesg(lmp, "Exceeded Selection Threshold. Current Value: {}\n", max_grade);
    if (max_grade > break_threshold)
      error->all(FLERR, "Exceeded Break Threshold. Terminating simulation.\n");
    ;
  }
}

/* ----------------------------------------------------------------------
   Extrapolation Calculation Function
------------------------------------------------------------------------- */
double PairMTPExtrapolation::calculate_extrapolation_grade(double *candidate_vector)
{
  // This should  use BLAS if possible
  double max_grade = 0;
  for (int i = 0; i < coeff_count; i++) {
    double current_grade = 0;
    for (int j = 0; j < coeff_count; j++) {
      current_grade += candidate_vector[j] * inverse_active_set[i][j];
    }
    max_grade = std::max(std::abs(current_grade), max_grade);
  }
  // This division by to ensure back-compatability. Not mathematical necesary.
  return max_grade / 2;
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairMTPExtrapolation::settings(int narg, char **arg)
{

  if (comm->me == 0) {
    if (narg < 4)
      error->all(
          FLERR,
          "Pair mtp/extrapolation only accepts 4 or 5 arguments: {potential_file} "
          "{extrapolation_mode} "
          "{selection_threshold} {break_threshold} {sampling_frequency, optional}. Currently "
          "specified: {} arguments!",
          narg);
    if (narg > 5)
      utils::logmesg(
          lmp,
          "Pair mtp/extrapolation only accepts 4 or 5 arguments: {potential_file} "
          "{extrapolation_mode} "
          // To avoid excessive copying and reduce memory footprint, we can set the pointer
          // for moment tensor vals to the appropriate index within energy_ders_wrt_coeffs
          "{selection_threshold} {break_threshold} {sampling_frequency, optional}. Ignoring "
          "excessive "
          "arguments!\n");
  }

  std::string mode_name = LAMMPS_NS::utils::lowercase(arg[1]);
  if (mode_name == "neighborhood" || mode_name == "neighbourhood")    //support for british spelling
    pool_grades = false;
  else if (mode_name == "configuration")
    pool_grades = true;
  else
    error->all(FLERR,
               "Only \"configuration\" and \"neighborhood\" extrapolation modes are supported. "
               "Currently specified: \"{}\"",
               arg[1]);

  select_threshold = utils::numeric(FLERR, arg[2], true, lmp);
  break_threshold = utils::numeric(FLERR, arg[3], true, lmp);
  if (narg == 5) sampling_frequency = utils::inumeric(FLERR, arg[4], true, lmp);

  if (comm->me == 0)
    utils::logmesg(
        lmp,
        "Sampling Scheme: {} mode, sampling every {} timestep(s) with a selection threshold of {} "
        "and break threshold of {}.\n",
        mode_name, sampling_frequency, select_threshold, break_threshold);

  FILE *mtp_file = utils::open_potential(arg[0], lmp, nullptr);
  read_file(mtp_file, arg[0]);
}

/* ----------------------------------------------------------------------
   MTP file parsing helper function. Includes memory allocation. Excludes some radial basis hyperparameters (in radial basis constructor instead).
------------------------------------------------------------------------- */
void PairMTPExtrapolation::read_file(FILE *mtp_file, char *file_path)
{
  PairMTP::read_file(mtp_file);

  // Some size calcs
  coeff_count = radial_coeff_count + species_count + alpha_scalar_count;
  int num_doubles = coeff_count * coeff_count;

  // Now we allocate memory for the additional memory needed for calculations
  memory->create(active_set, coeff_count, coeff_count, "active_set");
  memory->create(inverse_active_set, coeff_count, coeff_count, "inverse_active_set");
  memory->create(radial_moment_ders, alpha_moment_count, "radial_moment_ders");
  memory->create(energy_ders_wrt_coeffs, coeff_count, "energy_ders_wrt_coeffs");
  memory->create(radial_jacobian, alpha_index_basic_count, species_count,
                 radial_coeff_count_per_pair, "radial_jacobian");
  // We initialize the extrapolation grades during compute since its size depend on problem size.

  if (comm->me == 0) {
    std::string new_separators = "=, ";
    std::string separators = TOKENIZER_DEFAULT_SEPARATORS + new_separators;
    TextFileReader tfr(mtp_file, "ml-mtp");
    tfr.ignore_comments = false;

    // Read the weights. Not used but serves as a check.
    ValueTokenizer line_tokens = ValueTokenizer(std::string(tfr.next_line()), separators);
    std::string keyword = line_tokens.next_string();
    if (keyword != "#MVS_v1.1") {    // If there are no
      utils::logmesg(lmp,
                     "Untrained potential found. If the potential specified have been previously "
                     "trained, please verify that the MVS version is \"MVS_v1.1\". \n",
                     keyword);
      untrained_potential = true;
      return;
    }
    tfr.ignore_comments = true;    // Accept comments after reading the version which is a comment

    line_tokens = ValueTokenizer(std::string(tfr.next_line()), separators);
    keyword = line_tokens.next_string();
    if (keyword != "energy_weight")
      lmp->error->all(FLERR, "Error in reading MTP file, energy_weight");

    line_tokens = ValueTokenizer(std::string(tfr.next_line()), separators);
    keyword = line_tokens.next_string();
    if (keyword != "force_weight")
      lmp->error->all(FLERR, "Error in reading MTP file, force_weight");

    line_tokens = ValueTokenizer(std::string(tfr.next_line()), separators);
    keyword = line_tokens.next_string();
    if (keyword != "stress_weight")
      lmp->error->all(FLERR, "Error in reading MTP file, stress_weight");

    line_tokens = ValueTokenizer(std::string(tfr.next_line()), separators);
    keyword = line_tokens.next_string();
    if (keyword != "site_en_weight")
      lmp->error->all(FLERR, "Error in reading MTP file, site_en_weight");

    line_tokens = ValueTokenizer(std::string(tfr.next_line()), separators);
    keyword = line_tokens.next_string();
    if (keyword != "weight_scaling")
      lmp->error->all(FLERR, "Error in reading MTP file, weight_scaling");

    fgetc(mtp_file);    // We need to skip foward 1 character. There is a # before the binary data.
    utils::sfread(FLERR, &active_set[0][0], sizeof(double), num_doubles, mtp_file, nullptr,
                  lmp->error);
    utils::sfread(FLERR, &inverse_active_set[0][0], sizeof(double), num_doubles, mtp_file, nullptr,
                  lmp->error);
  }

  //Broadcast active set to others
  MPI_Bcast(&active_set[0][0], num_doubles, MPI_DOUBLE, 0, world);
  MPI_Bcast(&inverse_active_set[0][0], num_doubles, MPI_DOUBLE, 0, world);
}