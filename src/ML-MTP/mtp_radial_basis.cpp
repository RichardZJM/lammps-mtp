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

#include <algorithm>
#include <cmath>

using namespace LAMMPS_NS;

RadialMTPBasis::RadialMTPBasis(TextFileReader &tfr, LAMMPS *lmp) :
    Pointers(lmp), size(0), min_cutoff(0.0), max_cutoff(0.0), scaling(1.0),
    radial_basis_vals(nullptr), radial_basis_ders(nullptr)
{ read_basis_properties(tfr); }

RadialMTPBasis::RadialMTPBasis(int size, LAMMPS *lmp) :
    Pointers(lmp), size(size), min_cutoff(0.0), max_cutoff(0.0), scaling(1.0),
    radial_basis_vals(nullptr), radial_basis_ders(nullptr)
{
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
    error->all(FLERR, "Error in reading MTP file. Cannot read lower cutoff.");
  min_cutoff = line_tokens.next_double();

  // Read the upper cutoff
  line_tokens = ValueTokenizer(std::string(tfr.next_line()), separators);
  keyword = line_tokens.next_string();
  if (keyword != "max_val" && keyword != "max_dist")
    error->all(FLERR, "Error in reading MTP file. Cannot read upper cutoff.");
  max_cutoff = line_tokens.next_double();

  // Read the basis size set value
  line_tokens = ValueTokenizer(std::string(tfr.next_line()), separators);
  keyword = line_tokens.next_string();
  if (keyword != "radial_basis_size")
    error->all(FLERR, "Error in reading MTP file. Cannot read radial basis set size.");
  size = line_tokens.next_int();

  //Allocate the memory for the basis set values and derivatives.
  memory->create(radial_basis_vals, size, "pair:mtp_radial_vals");
  memory->create(radial_basis_ders, size, "pair:mtp_radial_ders");
}

RadialMTPBasis::~RadialMTPBasis()
{
  memory->destroy(radial_basis_vals);
  memory->destroy(radial_basis_ders);
}

// Default species-dependent variants: delegate to species-agnostic versions
void RadialMTPBasis::calc_radial_basis(double dist, int /*t1*/, int /*t2*/)
{ calc_radial_basis(dist); }

void RadialMTPBasis::calc_radial_basis_ders(double dist, int /*t1*/, int /*t2*/)
{ calc_radial_basis_ders(dist); }

// --- Fused and branch-free envelope helper functions (Ported from native LAMMPS version) ---

// Faster, branch-free smootherstep evaluation (6t^5 - 15t^4 + 10t^3)
static inline double smootherstep_val(double x, double min_val, double inv_range)
{
  double t = (x - min_val) * inv_range;
  t = std::max(0.0, std::min(1.0, t));
  return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

// // Branch-free derivative of the smootherstep function
// static inline double smootherstep_der(double x, double min_val, double inv_range)
// {
//   double t = (x - min_val) * inv_range;
//   double active = (t > 0.0) * (t < 1.0);
//   return active * 30.0 * t * t * (1.0 - t) * (1.0 - t) * inv_range;
// }

// Reversed smootherstep evaluation for the right cutoff (decays to 0 at cutoff)
static inline double rev_smootherstep_val(double x, double max_val, double inv_range)
{
  double t = (max_val - x) * inv_range;
  t = std::max(0.0, std::min(1.0, t));
  return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

// // Branch-free derivative of the reversed smootherstep function
// static inline double rev_smootherstep_der(double x, double max_val, double inv_range)
// {
//   double t = (max_val - x) * inv_range;
//   double active = (t > 0.0) * (t < 1.0);
//   return -active * 30.0 * t * t * (1.0 - t) * (1.0 - t) * inv_range;
// }

// Evaluates the product of the left and right smoothersteps
static inline double envelope(double val, double mindist, double cutoff, double inv_left_range,
                              double inv_right_range)
{
  double s_left = smootherstep_val(val, mindist, inv_left_range);
  double s_right = rev_smootherstep_val(val, cutoff, inv_right_range);
  return s_left * s_right;
}

// Dual-evaluation function to calculate both the envelope value and its derivative in one pass,
// reusing intermediate values (t_left, s_left, etc.) to minimize floating-point operations.
static inline void envelope_and_der(double val, double mindist, double cutoff,
                                    double inv_left_range, double inv_right_range, double &env,
                                    double &env_der)
{
  double t_left = (val - mindist) * inv_left_range;
  t_left = std::max(0.0, std::min(1.0, t_left));
  double s_left = t_left * t_left * t_left * (t_left * (t_left * 6.0 - 15.0) + 10.0);

  double t_right = (cutoff - val) * inv_right_range;
  t_right = std::max(0.0, std::min(1.0, t_right));
  double s_right = t_right * t_right * t_right * (t_right * (t_right * 6.0 - 15.0) + 10.0);

  env = s_left * s_right;

  double active_left = (t_left > 0.0) * (t_left < 1.0);
  double ds_left =
      active_left * 30.0 * t_left * t_left * (1.0 - t_left) * (1.0 - t_left) * inv_left_range;

  double active_right = (t_right > 0.0) * (t_right < 1.0);
  double ds_right = -active_right * 30.0 * t_right * t_right * (1.0 - t_right) * (1.0 - t_right) *
      inv_right_range;

  env_der = ds_left * s_right + s_left * ds_right;
}

void RBChebyshev::calc_radial_basis(double dist)
{
  double inv_range = 1.0 / (max_cutoff - min_cutoff);
  double ksi = (2.0 * dist - (min_cutoff + max_cutoff)) * inv_range;

  double d_max = dist - max_cutoff;
  double term = scaling * d_max * d_max;

  radial_basis_vals[0] = term;
  if (size > 1) { radial_basis_vals[1] = term * ksi; }

  double two_ksi = 2.0 * ksi;
  for (int i = 2; i < size; i++) {
    radial_basis_vals[i] = two_ksi * radial_basis_vals[i - 1] - radial_basis_vals[i - 2];
  }
}

void RBChebyshev::calc_radial_basis_ders(double dist)
{
  double inv_range = 1.0 / (max_cutoff - min_cutoff);
  double mult = 2.0 * inv_range;
  double ksi = (2.0 * dist - (min_cutoff + max_cutoff)) * inv_range;

  double d_max = dist - max_cutoff;
  double term = scaling * d_max * d_max;

  radial_basis_vals[0] = term;
  radial_basis_ders[0] = scaling * 2.0 * d_max;

  double two_ksi = 2.0 * ksi;
  if (size > 1) {
    radial_basis_vals[1] = term * ksi;
    radial_basis_ders[1] = scaling * d_max * (mult * d_max + two_ksi);
  }

  double two_mult = 2.0 * mult;
  for (int i = 2; i < size; i++) {
    radial_basis_vals[i] = two_ksi * radial_basis_vals[i - 1] - radial_basis_vals[i - 2];
    radial_basis_ders[i] = two_mult * radial_basis_vals[i - 1] +
        two_ksi * radial_basis_ders[i - 1] - radial_basis_ders[i - 2];
  }
}

// ----------------------------------------------------------------------
// LRBSChebyshev: LRBS Chebyshev with per-species-pair envelope
// ----------------------------------------------------------------------

// Proc 0: read envelope data and basis size from file
LRBSChebyshev::LRBSChebyshev(TextFileReader &tfr, LAMMPS *lmp, int species_count_in) :
    RadialMTPBasis(0, lmp), species_count(species_count_in), min_vals(nullptr), max_vals(nullptr),
    switching_points(nullptr), inv_left_ranges(nullptr), inv_right_ranges(nullptr),
    cheb_mults(nullptr), cheb_offsets(nullptr), precomputed(false)
{
  const int n2 = species_count * species_count;
  memory->create(min_vals, n2, "pair:lrbs_min_vals");
  memory->create(max_vals, n2, "pair:lrbs_max_vals");
  memory->create(switching_points, n2, "pair:lrbs_switching_points");
  memory->create(inv_left_ranges, n2, "pair:lrbs_inv_left_ranges");
  memory->create(inv_right_ranges, n2, "pair:lrbs_inv_right_ranges");
  memory->create(cheb_mults, n2, "pair:lrbs_cheb_mults");
  memory->create(cheb_offsets, n2, "pair:lrbs_cheb_offsets");

  const std::string new_separators = "=, ";
  const std::string separators = TOKENIZER_DEFAULT_SEPARATORS + new_separators;

  max_cutoff = 0.0;

  // Read lines: skip "radial_envelope" header, parse "t1-t2" then "{min, peak, max}"
  // on the next line, until "radial_basis_size" is found.
  while (true) {
    char *line = tfr.next_line();
    if (line == nullptr) error->one(FLERR, "Error reading LRBS_Chebyshev: unexpected end of file.");

    ValueTokenizer line_tokens(std::string(line), separators);
    if (!line_tokens.has_next()) continue;

    std::string first_token = line_tokens.next_string();

    // Check if this is the radial_basis_size line
    if (first_token == "radial_basis_size" || first_token == "basis_size") {
      size = line_tokens.next_int();
      if (size <= 0) error->one(FLERR, "Error reading LRBS_Chebyshev: invalid radial_basis_size.");
      break;
    }

    // Skip non-species-pair lines (e.g., "radial_envelope" header)
    if (first_token.find('-') == std::string::npos) continue;

    // Parse "t1-t2" from the first token
    std::size_t dash_pos = first_token.find('-');
    if (dash_pos == 0 || dash_pos == first_token.size() - 1) continue;

    int t1 = 0, t2 = 0;
    try {
      t1 = std::stoi(first_token.substr(0, dash_pos));
      t2 = std::stoi(first_token.substr(dash_pos + 1));
    } catch (...) {
      continue;    // Skip if it wasn't a valid species-pair line
    }

    if (t1 < 0 || t1 >= species_count || t2 < 0 || t2 >= species_count)
      error->one(FLERR, "Error reading LRBS_Chebyshev: species index out of range.");

    // Read the next line containing: {min, peak, max}
    char *param_line = tfr.next_line();
    if (param_line == nullptr)
      error->one(FLERR, "Error reading LRBS_Chebyshev: unexpected end of file in envelope data.");

    ValueTokenizer param_tokens(std::string(param_line), separators + "{}");
    if (!param_tokens.has_next())
      error->one(FLERR, "Error reading LRBS_Chebyshev: empty parameter line in envelope data.");

    double min_val = param_tokens.next_double();
    if (!param_tokens.has_next())
      error->one(FLERR, "Error reading LRBS_Chebyshev: missing peak in envelope data.");
    double spoint = param_tokens.next_double();
    if (!param_tokens.has_next())
      error->one(FLERR, "Error reading LRBS_Chebyshev: missing max_val in envelope data.");
    double max_val = param_tokens.next_double();

    if (min_val >= max_val)
      error->one(FLERR, "Error reading LRBS_Chebyshev: min_dist >= max_dist for pair {}-{}.", t1,
                 t2);
    if (spoint <= min_val || spoint >= max_val)
      error->one(
          FLERR,
          "Error reading LRBS_Chebyshev: peak must be strictly between min and max for pair {}-{}.",
          t1, t2);

    int idx = t1 * species_count + t2;
    min_vals[idx] = min_val;
    max_vals[idx] = max_val;
    switching_points[idx] = spoint;

    if (max_vals[idx] > max_cutoff) max_cutoff = max_vals[idx];
  }

  init_precomputations();

  // Allocate basis arrays now that we know the size
  memory->create(radial_basis_vals, size, "pair:mtp_radial_vals");
  memory->create(radial_basis_ders, size, "pair:mtp_radial_ders");
}

// Non-root: reconstruct from MPI broadcast data
LRBSChebyshev::LRBSChebyshev(int size_in, int species_count_in, LAMMPS *lmp) :
    RadialMTPBasis(size_in, lmp), species_count(species_count_in), min_vals(nullptr),
    max_vals(nullptr), switching_points(nullptr), inv_left_ranges(nullptr),
    inv_right_ranges(nullptr), cheb_mults(nullptr), cheb_offsets(nullptr), precomputed(false)
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

void LRBSChebyshev::init_precomputations()
{
  const int n2 = species_count * species_count;
  for (int i = 0; i < n2; i++) {
    double min_val = min_vals[i];
    double max_val = max_vals[i];
    double spoint = switching_points[i];

    if (spoint > min_val && spoint < max_val) {
      inv_left_ranges[i] = 1.0 / (spoint - min_val);
      inv_right_ranges[i] = 1.0 / (max_val - spoint);
      double total_range = max_val - min_val;
      cheb_mults[i] = 2.0 / total_range;
      cheb_offsets[i] = (min_val + max_val) / total_range;
    } else {
      inv_left_ranges[i] = 0.0;
      inv_right_ranges[i] = 0.0;
      cheb_mults[i] = 0.0;
      cheb_offsets[i] = 0.0;
    }
  }
  precomputed = true;
}

void LRBSChebyshev::calc_radial_basis(double dist, int t1, int t2)
{
  if (!precomputed) init_precomputations();

  int idx = t1 * species_count + t2;
  double min_val = min_vals[idx];
  double max_val = max_vals[idx];

  double inv_left = inv_left_ranges[idx];
  double inv_right = inv_right_ranges[idx];
  double mult = cheb_mults[idx];
  double offset = cheb_offsets[idx];

  double ksi = dist * mult - offset;
  double env = envelope(dist, min_val, max_val, inv_left, inv_right);

  radial_basis_vals[0] = scaling * env;
  if (size > 1) { radial_basis_vals[1] = scaling * env * ksi; }

  double two_ksi = 2.0 * ksi;
  for (int i = 2; i < size; i++) {
    radial_basis_vals[i] = two_ksi * radial_basis_vals[i - 1] - radial_basis_vals[i - 2];
  }
}

void LRBSChebyshev::calc_radial_basis_ders(double dist, int t1, int t2)
{
  if (!precomputed) init_precomputations();

  int idx = t1 * species_count + t2;
  double min_val = min_vals[idx];
  double max_val = max_vals[idx];

  double inv_left = inv_left_ranges[idx];
  double inv_right = inv_right_ranges[idx];
  double mult = cheb_mults[idx];
  double offset = cheb_offsets[idx];

  double ksi = dist * mult - offset;
  double env, env_der;
  envelope_and_der(dist, min_val, max_val, inv_left, inv_right, env, env_der);

  radial_basis_vals[0] = scaling * env;
  radial_basis_ders[0] = scaling * env_der;

  double two_ksi = 2.0 * ksi;
  double two_mult = 2.0 * mult;
  if (size > 1) {
    radial_basis_vals[1] = scaling * env * ksi;
    radial_basis_ders[1] = scaling * (env_der * ksi + env * mult);
  }

  for (int i = 2; i < size; i++) {
    radial_basis_vals[i] = two_ksi * radial_basis_vals[i - 1] - radial_basis_vals[i - 2];
    radial_basis_ders[i] = two_mult * radial_basis_vals[i - 1] +
        two_ksi * radial_basis_ders[i - 1] - radial_basis_ders[i - 2];
  }
}

void LRBSChebyshev::calc_radial_basis(double /*dist*/)
{
  error->all(FLERR,
             "Calling single-argument LRBSChebyshev::calc_radial_basis is not supported. Species "
             "pairs are required.");
}

void LRBSChebyshev::calc_radial_basis_ders(double /*dist*/)
{
  error->all(FLERR,
             "Calling single-argument LRBSChebyshev::calc_radial_basis_ders is not supported. "
             "Species pairs are required.");
}