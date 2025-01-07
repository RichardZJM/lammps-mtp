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
// Contributing author, Richard Meng, Queen's University at Kingston, 22.11.24, contact@richardzjm.com
//

#include "pair_mtp.h"

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
#include <cstring>

using namespace LAMMPS_NS;

PairMTP::PairMTP(LAMMPS *lmp) : Pair(lmp)
{
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;
}

/* ---------------------------------------------------------------------- */

PairMTP::~PairMTP()
{
  if (copymode) return;

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(moment_tensor_vals);
    memory->destroy(basis_vals);
    memory->destroy(basis_ders);
    memory->destroy(radial_basis_coeffs);
    memory->destroy(linear_coeffs);
    memory->destroy(species_coeffs);
    memory->destroy(alpha_index_basic);
    memory->destroy(alpha_index_times);
    memory->destroy(alpha_moment_mapping);
    delete radial_basis;
  }
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairMTP::settings(int narg, char **arg)
{
  if (narg != 1)
    error->all(FLERR, "Pair style MTP must have exact 1 argment, the MTP potential file name.");
  read_file(arg[0]);
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairMTP::coeff(int narg, char **arg)
{
  // The potential file is specified in the setting function instead.
  if (narg != 0) error->all(FLERR, "Only \"pair_coeff * *\n is permitted");
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairMTP::init_style()
{
  if (force->newton_pair == 0) error->all(FLERR, "Pair style MTP requires Newton Pair on");

  // Request a full neighbourhood list which is needed for MTP
  neighbor->add_request(this, NeighConst::REQ_FULL);
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairMTP::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR, "Not all pair coeffs are set");

  return rcutmax;
}

/* ----------------------------------------------------------------------
   MTP file parsing helper function. Includes  memory allocation. Excludes some radial basis hyperparameters (in radial basis constructor instead).
------------------------------------------------------------------------- */
void PairMTP::read_file(char *mtp_file_name)
{
  //Open the MTP file on proc 0
  if (comm->me == 0) {

    // Maybe the read section isn't needed and the potential file reader already handles errors.
    FILE *mtp_file;
    mtp_file = utils::open_potential(mtp_file_name, lmp, nullptr);
    if (mtp_file == nullptr)
      error->one(FLERR, "Cannot open MTP file {}: ", mtp_file_name, utils::getsyserror());

    PotentialFileReader pfr{lmp, mtp_file_name, "ml-mtp"};
    std::string separators = TOKENIZER_DEFAULT_SEPARATORS + '=,';

    ValueTokenizer line_tokens = ValueTokenizer(std::string(pfr.next_line()), separators);
    std::string keyword = line_tokens.next_string();

    if (keyword != "MTP")    // Version checking
      error->all(FLERR, "Only MTP potential files are accepted.");
    std::string version_line = std::string(pfr.next_line());
    if (version_line != "version = 1.1.0")    // Version checking
      error->all(FLERR, "MTP file must have version \"1.1.0\"");

    // Read the potential name (optional)
    line_tokens = ValueTokenizer(pfr.next_line(), separators);
    keyword = line_tokens.next_string();
    if (keyword != "potential_name") {
      try {
        potential_name = line_tokens.next_string();
      } catch (TokenizerException e) {
      }
      line_tokens = ValueTokenizer(pfr.next_line(), separators);
      keyword = line_tokens.next_string();
    }

    // Read the species count
    if (keyword != "species_count")
      error->all(FLERR, "Error reading MTP file. Species count not found.");
    species_count = line_tokens.next_int();

    // Read the potential tag (also optional)
    line_tokens = ValueTokenizer(pfr.next_line(), separators);
    keyword = line_tokens.next_string();
    if (keyword != "potential_tag") {
      try {
        potential_tag = line_tokens.next_string();
      } catch (TokenizerException e) {
      }
      line_tokens = ValueTokenizer(pfr.next_line(), separators);
      keyword = line_tokens.next_string();
    }

    // Read the radial basis type
    if (keyword != "radial_basis_type")
      error->all(FLERR, "Error reading MTP file. Not radial basis set type is specified.");
    std::string radial_basis_name = line_tokens.next_string();

    // Set the type of radial basis. No switch/case with strings...
    if (radial_basis_name == "RBChebyshev")
      radial_basis = new RBChebyshev(pfr, lmp);
    else
      error->all(FLERR,
                 "Error reading MTP file. The specified radial basis set type, {}, was not found..",
                 radial_basis_name);

    // Check for magnetic basis which is currently unsupported.
    line_tokens = ValueTokenizer(pfr.next_line(), separators);
    keyword = line_tokens.next_string();
    if (keyword != "radial_coeffs") {
      error->all(FLERR, "Initializing new potentials is currently not supported.");
    }

    // Get the radial function count
    line_tokens = ValueTokenizer(pfr.next_line(), separators);
    radial_func_count = line_tokens.next_int();

    // Allocate memory for radial basis
    int pairs_count = species_count * species_count;
    int radial_coeff_count_per_pair = radial_basis->size * radial_func_count;

    memory->create(radial_basis_coeffs, pairs_count * radial_coeff_count_per_pair,
                   "radial_basis_coeffs");

    // Read the radial basis coeffs
    for (int i = 0; i < pairs_count; i++) {
      //Read which pairs are being allocated
      line_tokens = ValueTokenizer(pfr.next_line(), separators + "-");
      int type1 = line_tokens.next_int();
      int type2 = line_tokens.next_int();
      setflag[type1][type2] = 1;    // Make sure the setflag is set

      // Read the coeffs for the pair. First find the offset in the array pointer.
      int pair_offset = type1 * species_count + type2;
      int offset = pair_offset * radial_coeff_count_per_pair;

      pfr.next_dvector(radial_basis_coeffs, radial_coeff_count_per_pair);
    }

    // Get the total alpha count
    line_tokens = ValueTokenizer(pfr.next_line(), separators);
    keyword = line_tokens.next_string();
    if (keyword != "alpha_moments_count")
      error->all(FLERR, "Error reading MTP file. Alpha moment count not found.");
    alpha_moment_count = line_tokens.next_int();

    // Get the basic alpha count
    line_tokens = ValueTokenizer(pfr.next_line(), separators);
    keyword = line_tokens.next_string();
    if (keyword != "alpha_index_basic_count")
      error->all(FLERR, "Error reading MTP file. Alpha moment count not found.");
    alpha_index_basic_count = line_tokens.next_int();

    // Read the basic alphas
    int radial_func_max = 0;
    line_tokens = ValueTokenizer(pfr.next_line(), separators + "{},");
    keyword = line_tokens.next_string();
    if (keyword != "alpha_index_basic")
      error->all(FLERR, "Error reading MTP file. Alpha index basic not found.");
    memory->create(alpha_index_basic, alpha_index_basic_count, "alpha_index_basic");
    for (int i = 0; i < alpha_index_basic_count; i++) {
      for (int j = 0; j < 4; j++) { alpha_index_basic[i][j] = line_tokens.next_int(); }
      if (alpha_index_basic[i][0] > radial_func_max) radial_func_max = alpha_index_basic[i][0];
    }
    if (radial_func_max != radial_func_count - 1)    //Index validity check
      error->all(FLERR, "Wrong number of radial functions specified!");

    // Get the alpha times count
    line_tokens = ValueTokenizer(pfr.next_line(), separators);
    keyword = line_tokens.next_string();
    if (keyword != "alpha_index_times_count")
      error->all(FLERR, "Error reading MTP file. Alpha index times count not found.");
    alpha_index_times_count = line_tokens.next_int();

    // Read the alphas times
    line_tokens = ValueTokenizer(pfr.next_line(), separators + "{},");
    keyword = line_tokens.next_string();
    if (keyword != "alpha_index_times")
      error->all(FLERR, "Error reading MTP file. Alpha index times not found.");
    memory->create(alpha_index_times, alpha_index_times_count, "alpha_index_times");
    for (int i = 0; i < alpha_index_times_count; i++) {
      for (int j = 0; j < 4; j++) { alpha_index_times[i][j] = line_tokens.next_int(); }
    }

    // Get the alpha scalar count
    line_tokens = ValueTokenizer(pfr.next_line(), separators);
    keyword = line_tokens.next_string();
    if (keyword != "alpha_scalar_moments")
      error->all(FLERR, "Error reading MTP file. Alpha scalar moment count not found.");
    alpha_scalar_count = line_tokens.next_int();

    //Read the alpha moment mappings
    line_tokens = ValueTokenizer(pfr.next_line(), separators + "{},");
    keyword = line_tokens.next_string();
    if (keyword != "alpha_moment_mapping")
      error->all(FLERR, "Error reading MTP file. Alpha moment mappings not found.");
    memory->create(alpha_moment_mapping, alpha_scalar_count, "alpha_moment_mapping");
    for (int i = 0; i < alpha_scalar_count; i++) {
      alpha_moment_mapping[i] = line_tokens.next_int();
    }
    // alpha_scalar_count++;    // The 0th rank tensor scalar is accounted for by the species coefficients

    //Read the species coefficients
    line_tokens = ValueTokenizer(pfr.next_line(), separators + "{},");
    keyword = line_tokens.next_string();
    if (keyword != "species_coeffs")
      error->all(FLERR, "Error reading MTP file. Species coefficients not found.");
    memory->create(species_coeffs, species_count, "species_coeffs");
    for (int i = 0; i < species_count; i++) { species_coeffs[i] = line_tokens.next_double(); }

    //Read the alpha moment mappings
    line_tokens = ValueTokenizer(pfr.next_line(), separators + "{},");
    keyword = line_tokens.next_string();
    if (keyword != "alpha_moment_mapping")
      error->all(FLERR, "Error reading MTP file. Alpha moment mappings not found.");
    memory->create(alpha_moment_mapping, alpha_scalar_count, "alpha_moment_mapping");
    for (int i = 0; i < alpha_scalar_count; i++) {
      alpha_moment_mapping[i] = line_tokens.next_int();
    }

    //Read the linear MTP basis coefficients
    line_tokens = ValueTokenizer(pfr.next_line(), separators + "{},");
    keyword = line_tokens.next_string();
    if (keyword != "moment_coeffs")
      error->all(FLERR, "Error reading MTP file. Moment coefficients not found.");
    memory->create(linear_coeffs, alpha_scalar_count, "moment_coeffs");
    for (int i = 0; i < alpha_scalar_count; i++) { linear_coeffs[i] = line_tokens.next_double(); }
  }
}