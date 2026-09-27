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

#include "mtp_radial_basis.h"

#include "error.h"
#include "memory.h"
#include "text_file_reader.h"
#include "utils.h"

#include <algorithm>

using namespace LAMMPS_NS;

RadialMTPBasis::RadialMTPBasis(TextFileReader &tfr, LAMMPS *lmp) :
    Pointers(lmp), size(0), min_cutoff(0.0), max_cutoff(0.0), scaling(1.0),
    radial_basis_vals(nullptr), radial_basis_ders(nullptr)
{ read_basis_properties(tfr); }

RadialMTPBasis::RadialMTPBasis(LAMMPS *lmp) :
    Pointers(lmp), size(0), min_cutoff(0.0), max_cutoff(0.0), scaling(1.0),
    radial_basis_vals(nullptr), radial_basis_ders(nullptr)
{
}

RadialMTPBasis::RadialMTPBasis(int size, LAMMPS *lmp) :
    Pointers(lmp), size(size), min_cutoff(0.0), max_cutoff(0.0), scaling(1.0),
    radial_basis_vals(nullptr), radial_basis_ders(nullptr)
{
  if (size < 2) error->one(FLERR, "MTP radial basis size must be at least 2.");
  memory->create(radial_basis_vals, size, "pair:mtp_radial_vals");
  memory->create(radial_basis_ders, size, "pair:mtp_radial_ders");
}

void RadialMTPBasis::read_basis_properties(TextFileReader &tfr)
{
  const std::string new_separators = "=, ";
  const std::string separators = TOKENIZER_DEFAULT_SEPARATORS + new_separators;

  ValueTokenizer line_tokens{std::string(tfr.next_line()), separators};
  std::string keyword = line_tokens.next_string();

  // First check if scaling is available
  if (keyword == "scaling") {
    scaling = line_tokens.next_double();
    line_tokens = ValueTokenizer(std::string(tfr.next_line()), separators);
    keyword = line_tokens.next_string();
  }

  // Read the lower cutoff
  if (keyword != "min_val" && keyword != "min_dist")
    error->one(FLERR, "Error in reading MTP file. Cannot read lower cutoff.");
  min_cutoff = line_tokens.next_double();

  // Read the upper cutoff
  line_tokens = ValueTokenizer(std::string(tfr.next_line()), separators);
  keyword = line_tokens.next_string();
  if (keyword != "max_val" && keyword != "max_dist")
    error->one(FLERR, "Error in reading MTP file. Cannot read upper cutoff.");
  max_cutoff = line_tokens.next_double();
  if (!(max_cutoff > min_cutoff))
    error->one(FLERR, "MTP maximum cutoff must exceed the minimum cutoff.");

  // Read the basis size set value
  line_tokens = ValueTokenizer(std::string(tfr.next_line()), separators);
  keyword = line_tokens.next_string();
  if (keyword != "radial_basis_size")
    error->one(FLERR, "Error in reading MTP file. Cannot read radial basis set size.");
  size = line_tokens.next_int();
  if (size < 2) error->one(FLERR, "MTP radial basis size must be at least 2.");

  //Allocate the memory for the basis set values and derivatives.
  memory->create(radial_basis_vals, size, "pair:mtp_radial_vals");
  memory->create(radial_basis_ders, size, "pair:mtp_radial_ders");
}

RadialMTPBasis::~RadialMTPBasis()
{
  memory->destroy(radial_basis_vals);
  memory->destroy(radial_basis_ders);
}

void RBChebyshev::calc_radial_basis_ders(double dist, int /*itype*/, int /*jtype*/)
{
  const double delta = dist - max_cutoff;
  const double span = max_cutoff - min_cutoff;
  const double mult = 2.0 / span;
  const double ksi = (2 * dist - (min_cutoff + max_cutoff)) / span;

  double val0 = scaling * delta * delta;
  double val1 = scaling * (ksi * delta * delta);
  double der0 = scaling * 2 * delta;
  double der1 = scaling * (mult * delta * delta + 2 * ksi * delta);
  radial_basis_vals[0] = val0;
  radial_basis_vals[1] = val1;
  radial_basis_ders[0] = der0;
  radial_basis_ders[1] = der1;

  for (int i = 2; i < size; i++) {
    const double val = 2 * ksi * val1 - val0;
    const double der = 2 * (mult * val1 + ksi * der1) - der0;
    radial_basis_vals[i] = val;
    radial_basis_ders[i] = der;
    val0 = val1;
    val1 = val;
    der0 = der1;
    der1 = der;
  }
}

// ----------------------------------------------------------------------
// LRBSChebyshev: Chebyshev basis with a per-species-pair envelope
// ----------------------------------------------------------------------

// Proc 0: read the envelope data and the basis size from the file
LRBSChebyshev::LRBSChebyshev(TextFileReader &tfr, LAMMPS *lmp, int species_count_in) :
    RadialMTPBasis(lmp), species_count(species_count_in), min_vals(nullptr), max_vals(nullptr),
    switching_points(nullptr), inv_left_ranges(nullptr), inv_right_ranges(nullptr),
    cheb_mults(nullptr), cheb_offsets(nullptr)
{
  allocate_pair_arrays();
  const int n2 = species_count * species_count;
  std::fill(min_vals, min_vals + n2, 0.0);
  std::fill(max_vals, max_vals + n2, 0.0);
  std::fill(switching_points, switching_points + n2, 0.0);

  const std::string new_separators = "=, ";
  const std::string separators = TOKENIZER_DEFAULT_SEPARATORS + new_separators;

  // Read lines: skip the "radial_envelope" header, parse "t1-t2" and then "{min, peak, max}"
  // on the next line, until "radial_basis_size" is found.
  while (true) {
    char *line = tfr.next_line();
    if (line == nullptr) error->one(FLERR, "Error reading LRBSChebyshev: unexpected end of file.");

    ValueTokenizer line_tokens(std::string(line), separators);
    if (!line_tokens.has_next()) continue;

    std::string first_token = line_tokens.next_string();

    if ((first_token == "radial_basis_size") || (first_token == "basis_size")) {
      size = line_tokens.next_int();
      if (size < 2) error->one(FLERR, "MTP radial basis size must be at least 2.");
      break;
    }

    // Skip anything that is not a species-pair line (e.g. the "radial_envelope" header)
    const std::size_t dash_pos = first_token.find('-');
    if (dash_pos == std::string::npos) continue;
    const std::string t1_str = first_token.substr(0, dash_pos);
    const std::string t2_str = first_token.substr(dash_pos + 1);
    if (!utils::is_integer(t1_str) || !utils::is_integer(t2_str)) continue;

    const int t1 = std::stoi(t1_str);
    const int t2 = std::stoi(t2_str);
    if ((t1 < 0) || (t1 >= species_count) || (t2 < 0) || (t2 >= species_count))
      error->one(FLERR, "Error reading LRBSChebyshev: species index out of range.");

    // The next line contains: {min, peak, max}
    char *param_line = tfr.next_line();
    if (param_line == nullptr)
      error->one(FLERR, "Error reading LRBSChebyshev: unexpected end of file in envelope data.");

    ValueTokenizer param_tokens(std::string(param_line), separators + "{}");
    if (param_tokens.count() < 3)
      error->one(FLERR, "Error reading LRBSChebyshev: expected {{min, peak, max}} for pair {}-{}.",
                 t1, t2);
    const double min_val = param_tokens.next_double();
    const double spoint = param_tokens.next_double();
    const double max_val = param_tokens.next_double();

    if (min_val >= max_val)
      error->one(FLERR, "Error reading LRBSChebyshev: min_dist >= max_dist for pair {}-{}.", t1,
                 t2);
    if ((spoint <= min_val) || (spoint >= max_val))
      error->one(
          FLERR,
          "Error reading LRBSChebyshev: peak must be strictly between min and max for pair {}-{}.",
          t1, t2);

    const int idx = t1 * species_count + t2;
    min_vals[idx] = min_val;
    max_vals[idx] = max_val;
    switching_points[idx] = spoint;

    if (max_val > max_cutoff) max_cutoff = max_val;
  }

  if (!(max_cutoff > 0.0))
    error->one(FLERR, "Error reading LRBSChebyshev: no envelope data found.");

  memory->create(radial_basis_vals, size, "pair:mtp_radial_vals");
  memory->create(radial_basis_ders, size, "pair:mtp_radial_ders");
}

// Non-root procs: the pair arrays are filled by MPI broadcast afterwards
LRBSChebyshev::LRBSChebyshev(int size_in, int species_count_in, LAMMPS *lmp) :
    RadialMTPBasis(size_in, lmp), species_count(species_count_in), min_vals(nullptr),
    max_vals(nullptr), switching_points(nullptr), inv_left_ranges(nullptr),
    inv_right_ranges(nullptr), cheb_mults(nullptr), cheb_offsets(nullptr)
{
  allocate_pair_arrays();
}

LRBSChebyshev::~LRBSChebyshev()
{
  memory->destroy(min_vals);
  memory->destroy(max_vals);
  memory->destroy(switching_points);
  memory->destroy(inv_left_ranges);
  memory->destroy(inv_right_ranges);
  memory->destroy(cheb_mults);
  memory->destroy(cheb_offsets);
}

void LRBSChebyshev::allocate_pair_arrays()
{
  const int n2 = species_count * species_count;
  memory->create(min_vals, n2, "pair:lrbs_min_vals");
  memory->create(max_vals, n2, "pair:lrbs_max_vals");
  memory->create(switching_points, n2, "pair:lrbs_switching_points");
  memory->create(inv_left_ranges, n2, "pair:lrbs_inv_left_ranges");
  memory->create(inv_right_ranges, n2, "pair:lrbs_inv_right_ranges");
  memory->create(cheb_mults, n2, "pair:lrbs_cheb_mults");
  memory->create(cheb_offsets, n2, "pair:lrbs_cheb_offsets");
}

// Pairs without envelope data get all-zero parameters, so their basis vanishes
void LRBSChebyshev::init_precomputations()
{
  const int n2 = species_count * species_count;
  for (int i = 0; i < n2; i++) {
    const double min_val = min_vals[i];
    const double max_val = max_vals[i];
    const double spoint = switching_points[i];

    if ((spoint > min_val) && (spoint < max_val)) {
      inv_left_ranges[i] = 1.0 / (spoint - min_val);
      inv_right_ranges[i] = 1.0 / (max_val - spoint);
      const double total_range = max_val - min_val;
      cheb_mults[i] = 2.0 / total_range;
      cheb_offsets[i] = (min_val + max_val) / total_range;
    } else {
      inv_left_ranges[i] = 0.0;
      inv_right_ranges[i] = 0.0;
      cheb_mults[i] = 0.0;
      cheb_offsets[i] = 0.0;
    }
  }
}

// Product of a rising smootherstep (min -> peak) and a falling one (peak -> max), and its
// derivative, sharing the intermediate values.
static inline void envelope_and_der(double val, double mindist, double cutoff,
                                    double inv_left_range, double inv_right_range, double &env,
                                    double &env_der)
{
  double t_left = (val - mindist) * inv_left_range;
  t_left = std::max(0.0, std::min(1.0, t_left));
  const double s_left = t_left * t_left * t_left * (t_left * (t_left * 6.0 - 15.0) + 10.0);

  double t_right = (cutoff - val) * inv_right_range;
  t_right = std::max(0.0, std::min(1.0, t_right));
  const double s_right = t_right * t_right * t_right * (t_right * (t_right * 6.0 - 15.0) + 10.0);

  env = s_left * s_right;

  const double active_left = (t_left > 0.0) * (t_left < 1.0);
  const double ds_left =
      active_left * 30.0 * t_left * t_left * (1.0 - t_left) * (1.0 - t_left) * inv_left_range;

  const double active_right = (t_right > 0.0) * (t_right < 1.0);
  const double ds_right = -active_right * 30.0 * t_right * t_right * (1.0 - t_right) *
      (1.0 - t_right) * inv_right_range;

  env_der = ds_left * s_right + s_left * ds_right;
}

void LRBSChebyshev::calc_radial_basis_ders(double dist, int itype, int jtype)
{
  const int idx = itype * species_count + jtype;
  const double mult = cheb_mults[idx];
  const double ksi = dist * mult - cheb_offsets[idx];

  double env, env_der;
  envelope_and_der(dist, min_vals[idx], max_vals[idx], inv_left_ranges[idx], inv_right_ranges[idx],
                   env, env_der);

  double val0 = scaling * env;
  double val1 = val0 * ksi;
  double der0 = scaling * env_der;
  double der1 = scaling * (env_der * ksi + env * mult);
  radial_basis_vals[0] = val0;
  radial_basis_vals[1] = val1;
  radial_basis_ders[0] = der0;
  radial_basis_ders[1] = der1;

  for (int i = 2; i < size; i++) {
    const double val = 2 * ksi * val1 - val0;
    const double der = 2 * (mult * val1 + ksi * der1) - der0;
    radial_basis_vals[i] = val;
    radial_basis_ders[i] = der;
    val0 = val1;
    val1 = val;
    der0 = der1;
    der1 = der;
  }
}
