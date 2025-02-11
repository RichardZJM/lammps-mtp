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

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairMTPExtrapolation::~PairMTPExtrapolation()
{
  PairMTP::~PairMTP();
  if (allocated) {
    memory->destroy(active_set);
    memory->destroy(inverse_active_set);
    memory->destroy(extrapolation_grades);
    memory->destroy(radial_jacobian);
    memory->destroy(radial_basic_ders);
    memory->destroy(radial_moment_ders);
    memory->destroy(basis_ders_wrt_coeffs);
  }
}

/* ----------------------------------------------------------------------
   Straightfoward MTP implementation based on MLIP3
   ---------------------------------------------------------------------- */

void PairMTPExtrapolation::compute(int eflag, int vflag)
{

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

    if (jac_size < jnum) {
      memory->grow(moment_jacobian, alpha_index_basic_count, jnum, 3,
                   "moment_jacobian");    // Resize the working jacobian.
      jac_size = jnum;
    }
    std::fill(&moment_tensor_vals[0], &moment_tensor_vals[0] + alpha_moment_count,
              0.0);    //Fill moments with 0
    std::fill(&nbh_energy_ders_wrt_moments[0], &nbh_energy_ders_wrt_moments[0] + alpha_moment_count,
              0.0);    //Fill moment derivatives with 0

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

      const double dist_sq = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];

      if (dist_sq > cutsq[itype + 1][jtype + 1]) continue;    //1 indexing

      const double dist = std::sqrt(dist_sq);
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

        //Find the offset for the radial basis coeffs
        int pair_offset = itype * species_count + jtype;
        int offset = (pair_offset * radial_basis_size * radial_func_count) + mu * radial_basis_size;

        // Find the radial component and its derivative
        for (int ri = 0; ri < radial_basis_size; ri++) {
          val += radial_basis_coeffs[offset + ri] * radial_basis->radial_basis_vals[ri];
          der += radial_basis_coeffs[offset + ri] * radial_basis->radial_basis_ders[ri];
        }

        // Normalize by the rank of alpha's coresponding tensor
        int norm_rank = alpha_index_basic[k][1] + alpha_index_basic[k][2] + alpha_index_basic[k][3];
        double norm_fac = 1.0 / dist_powers[norm_rank];
        val *= norm_fac;
        der = der * norm_fac - norm_rank * val / dist;

        double pow0 = coord_powers[alpha_index_basic[k][1]][0];
        double pow1 = coord_powers[alpha_index_basic[k][2]][1];
        double pow2 = coord_powers[alpha_index_basic[k][3]][2];
        double pow = pow0 * pow1 * pow2;
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
    if (eflag_atom || eflag_global) {        // This could be replaced with eflag_either
      nbh_energy = species_coeffs[itype];    // Essentially the reference point energy per species
      for (int k = 0; k < alpha_scalar_count; k++)
        nbh_energy += linear_coeffs[k] * moment_tensor_vals[alpha_moment_mapping[k]];

      // Tally energies per flags
      if (eflag_atom) eatom[i] = nbh_energy;
      if (eflag_global) eng_vdwl += nbh_energy;
    }

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

    //------------ Step 3: Multiply energy ders wrt moment by the Jacobian to get forces ------------
    for (int jj = 0; jj < jnum; jj++) {
      int j = firstneigh[i][jj];
      j &= NEIGHMASK;

      double r[3] = {x[j][0] - xi[0], x[j][1] - xi[1], x[j][2] - xi[2]};
      double rsq = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
      if (rsq > max_cutoff_sq) continue;

      double temp_force[3] = {0, 0, 0};
      for (int k = 0; k < alpha_index_basic_count; k++)
        for (int a = 0; a < 3; a++) {
          //Calculate forces
          temp_force[a] += nbh_energy_ders_wrt_moments[k] * moment_jacobian[k][jj][a];
        }

      f[i][0] += temp_force[0];
      f[i][1] += temp_force[1];
      f[i][2] += temp_force[2];

      f[j][0] -= temp_force[0];
      f[j][1] -= temp_force[1];
      f[j][2] -= temp_force[2];

      //Calculate virial stress
      if (vflag) {
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
  }
}
/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairMTPExtrapolation::settings(int narg, char **arg)
{
  if (narg != 3 && comm->me == 0)
    utils::logmesg(lmp,
                   "Pair MTP only accepts 3 arguments: {potential_file} {selection_threshold} "
                   "{break_threshold}. Ignoring excessive arguments!\n");
  FILE *mtp_file = utils::open_potential(arg[0], lmp, nullptr);

  read_file(mtp_file);
}

/* ----------------------------------------------------------------------
   MTP file parsing helper function. Includes memory allocation. Excludes some radial basis hyperparameters (in radial basis constructor instead).
------------------------------------------------------------------------- */
void PairMTPExtrapolation::read_file(FILE *mtp_file)
{
  PairMTP::read_file(mtp_file);

  // Now we allocate memory for the active set and its inverse
  int pairs_count = species_count * species_count;
  int radial_coeff_count_per_pair = radial_basis_size * radial_func_count;
  int radial_coeff_count = pairs_count * radial_coeff_count_per_pair;
  coeff_count = radial_coeff_count + species_count + alpha_scalar_count;
  memory->create(active_set, coeff_count, coeff_count, "pair:active_set");
  memory->create(inverse_active_set, coeff_count, coeff_count, "pair:inverse_active_set");

  if (comm->me == 0) {
    std::string new_separators = "=, ";
    std::string separators = TOKENIZER_DEFAULT_SEPARATORS + new_separators;
    TextFileReader tfr(mtp_file, "ml-mtp");
    tfr.ignore_comments = true;

    // Read the weights. Not used but serves as a check
    {
      ValueTokenizer line_tokens = ValueTokenizer(std::string(tfr.next_line()), separators);
      std::string keyword = line_tokens.next_string();
      if (keyword != "#MVS_v1.1")
        lmp->error->all(FLERR, "Error in reading MTP file. Wrong MVS version!");

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
    }

    // Read the active set and its inverse
    // It is store as a binary file so we need to
  }
}