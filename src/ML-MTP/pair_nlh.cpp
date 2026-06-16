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

#include "pair_nlh.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "utils.h"

#include <cmath>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairNLH::PairNLH(LAMMPS *lmp) : Pair(lmp)
{ writedata = 1; }

/* ---------------------------------------------------------------------- */

PairNLH::~PairNLH()
{
  if (copymode) return;

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    memory->destroy(z);
    memory->destroy(a1);
    memory->destroy(a2);
    memory->destroy(a3);
    memory->destroy(b1);
    memory->destroy(b2);
    memory->destroy(b3);
    memory->destroy(r1);
    memory->destroy(r2);
    memory->destroy(r1sq);
    memory->destroy(r2sq);
    memory->destroy(zze);
  }
}

/* ---------------------------------------------------------------------- */

void PairNLH::compute(int eflag, int vflag)
{
  int i, j, ii, jj, inum, jnum, itype, jtype;
  double xtmp, ytmp, ztmp, delx, dely, delz, evdwl, fpair;
  double rsq, r;
  int *ilist, *jlist, *numneigh, **firstneigh;

  evdwl = 0.0;
  ev_init(eflag, vflag);

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int newton_pair = force->newton_pair;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx * delx + dely * dely + delz * delz;
      jtype = type[j];

      if (rsq < r2sq[itype][jtype]) {
        r = sqrt(rsq);
        double v, dvdr;
        eval_nlh(r, itype, jtype, v, dvdr);

        // Force prefactor: fpair = -1/r * dV/dr
        fpair = -dvdr / r;

        f[i][0] += delx * fpair;
        f[i][1] += dely * fpair;
        f[i][2] += delz * fpair;
        if (newton_pair || j < nlocal) {
          f[j][0] -= delx * fpair;
          f[j][1] -= dely * fpair;
          f[j][2] -= delz * fpair;
        }

        if (eflag) { evdwl = v; }

        if (evflag) ev_tally(i, j, nlocal, newton_pair, evdwl, 0.0, fpair, delx, dely, delz);
      }
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ---------------------------------------------------------------------- */

void PairNLH::allocate()
{
  allocated = 1;
  const int np1 = atom->ntypes + 1;

  memory->create(setflag, np1, np1, "pair:setflag");
  for (int i = 1; i < np1; i++)
    for (int j = i; j < np1; j++) setflag[i][j] = 0;

  memory->create(cutsq, np1, np1, "pair:cutsq");

  memory->create(z, np1, "pair:z");
  memory->create(a1, np1, np1, "pair:a1");
  memory->create(a2, np1, np1, "pair:a2");
  memory->create(a3, np1, np1, "pair:a3");
  memory->create(b1, np1, np1, "pair:b1");
  memory->create(b2, np1, np1, "pair:b2");
  memory->create(b3, np1, np1, "pair:b3");
  memory->create(r1, np1, np1, "pair:r1");
  memory->create(r2, np1, np1, "pair:r2");
  memory->create(r1sq, np1, np1, "pair:r1sq");
  memory->create(r2sq, np1, np1, "pair:r2sq");
  memory->create(zze, np1, np1, "pair:zze");
}

/* ---------------------------------------------------------------------- */

void PairNLH::settings(int narg, char **arg)
{
  if (narg != 0) error->all(FLERR, "Illegal pair_style command");
}

/* ---------------------------------------------------------------------- */

void PairNLH::coeff(int narg, char **arg)
{
  double z_one, z_two;
  double a1_val, a2_val, a3_val;
  double b1_val, b2_val, b3_val;
  double r1_val, r2_val;

  if (narg != 12) error->all(FLERR, "Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo, ihi;
  utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);

  int jlo, jhi;
  utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);

  z_one = utils::numeric(FLERR, arg[2], false, lmp);
  z_two = utils::numeric(FLERR, arg[3], false, lmp);
  a1_val = utils::numeric(FLERR, arg[4], false, lmp);
  a2_val = utils::numeric(FLERR, arg[5], false, lmp);
  a3_val = utils::numeric(FLERR, arg[6], false, lmp);
  b1_val = utils::numeric(FLERR, arg[7], false, lmp);
  b2_val = utils::numeric(FLERR, arg[8], false, lmp);
  b3_val = utils::numeric(FLERR, arg[9], false, lmp);
  r1_val = utils::numeric(FLERR, arg[10], false, lmp);
  r2_val = utils::numeric(FLERR, arg[11], false, lmp);

  if (r1_val <= 0.0 || r2_val <= r1_val)
    error->all(FLERR, "Incorrect cutoffs for pair coefficients");

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo, i); j <= jhi; j++) {
      if (i == j) {
        if (z_one != z_two) error->all(FLERR, "Incorrect args for pair coefficients");
        z[i] = z_one;
      }
      setflag[i][j] = 1;
      set_coeff(i, j, z_one, z_two, a1_val, a2_val, a3_val, b1_val, b2_val, b3_val, r1_val, r2_val);
      count++;
    }
  }

  if (count == 0) error->all(FLERR, "Incorrect args for pair coefficients");
}

/* ---------------------------------------------------------------------- */

void PairNLH::init_style()
{ neighbor->add_request(this); }

/* ---------------------------------------------------------------------- */

double PairNLH::init_one(int i, int j)
{
  if (setflag[i][j] == 0) { error->all(FLERR, "All pair coeffs are not set in pair_style nlh"); }
  return r2[i][j];
}

/* ---------------------------------------------------------------------- */

void PairNLH::write_restart(FILE *fp)
{
  int i, j;
  for (i = 1; i <= atom->ntypes; i++) { fwrite(&z[i], sizeof(double), 1, fp); }
  for (i = 1; i <= atom->ntypes; i++) {
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j], sizeof(int), 1, fp);
      if (setflag[i][j]) {
        fwrite(&a1[i][j], sizeof(double), 1, fp);
        fwrite(&a2[i][j], sizeof(double), 1, fp);
        fwrite(&a3[i][j], sizeof(double), 1, fp);
        fwrite(&b1[i][j], sizeof(double), 1, fp);
        fwrite(&b2[i][j], sizeof(double), 1, fp);
        fwrite(&b3[i][j], sizeof(double), 1, fp);
        fwrite(&r1[i][j], sizeof(double), 1, fp);
        fwrite(&r2[i][j], sizeof(double), 1, fp);
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void PairNLH::read_restart(FILE *fp)
{
  allocate();

  int i, j;
  int me = comm->me;
  for (i = 1; i <= atom->ntypes; i++) {
    if (me == 0) utils::sfread(FLERR, &z[i], sizeof(double), 1, fp, nullptr, error);
    MPI_Bcast(&z[i], 1, MPI_DOUBLE, 0, world);
  }

  for (i = 1; i <= atom->ntypes; i++) {
    for (j = i; j <= atom->ntypes; j++) {
      if (me == 0) utils::sfread(FLERR, &setflag[i][j], sizeof(int), 1, fp, nullptr, error);
      MPI_Bcast(&setflag[i][j], 1, MPI_INT, 0, world);
      if (setflag[i][j]) {
        double a1_val, a2_val, a3_val;
        double b1_val, b2_val, b3_val;
        double r1_val, r2_val;
        if (me == 0) {
          utils::sfread(FLERR, &a1_val, sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &a2_val, sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &a3_val, sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &b1_val, sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &b2_val, sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &b3_val, sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &r1_val, sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &r2_val, sizeof(double), 1, fp, nullptr, error);
        }
        MPI_Bcast(&a1_val, 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&a2_val, 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&a3_val, 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&b1_val, 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&b2_val, 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&b3_val, 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&r1_val, 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&r2_val, 1, MPI_DOUBLE, 0, world);

        set_coeff(i, j, z[i], z[j], a1_val, a2_val, a3_val, b1_val, b2_val, b3_val, r1_val, r2_val);
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void PairNLH::write_restart_settings(FILE *) {}
void PairNLH::read_restart_settings(FILE *) {}

/* ---------------------------------------------------------------------- */

void PairNLH::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++) {
    fprintf(fp, "%d %g %g %g %g %g %g %g %g %g %g\n", i, z[i], z[i], a1[i][i], a2[i][i], a3[i][i],
            b1[i][i], b2[i][i], b3[i][i], r1[i][i], r2[i][i]);
  }
}

/* ---------------------------------------------------------------------- */

void PairNLH::write_data_all(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++) {
    for (int j = i; j <= atom->ntypes; j++) {
      fprintf(fp, "%d %d %g %g %g %g %g %g %g %g %g %g %g\n", i, j, z[i], z[j], a1[i][j], a2[i][j],
              a3[i][j], b1[i][j], b2[i][j], b3[i][j], r1[i][j], r2[i][j]);
    }
  }
}

/* ---------------------------------------------------------------------- */

double PairNLH::single(int /*i*/, int /*j*/, int itype, int jtype, double rsq, double /*dummy1*/,
                       double /*dummy2*/, double &fforce)
{
  double r = sqrt(rsq);
  double v, dvdr;
  eval_nlh(r, itype, jtype, v, dvdr);
  fforce = -dvdr / r;
  return v;
}

/* ----------------------------------------------------------------------
   evaluate energy and force derivative analytically
------------------------------------------------------------------------- */

void PairNLH::eval_nlh(double r, int i, int j, double &v, double &dvdr)
{
  double rinv = 1.0 / r;
  double e1 = exp(-b1[i][j] * r);
  double e2 = exp(-b2[i][j] * r);
  double e3 = exp(-b3[i][j] * r);

  double sum = a1[i][j] * e1 + a2[i][j] * e2 + a3[i][j] * e3;
  double sum_p = -a1[i][j] * b1[i][j] * e1 - a2[i][j] * b2[i][j] * e2 - a3[i][j] * b3[i][j] * e3;

  double E = zze[i][j] * sum * rinv;
  double dEdr = zze[i][j] * (sum_p - sum * rinv) * rinv;

  double r1_ij = r1[i][j];
  double r2_ij = r2[i][j];

  if (r <= r1_ij) {
    v = E;
    dvdr = dEdr;
  } else if (r < r2_ij) {
    double w = 1.0 / (r2_ij - r1_ij);
    double x = (r - r1_ij) * w;

    // Smooth 5th-order polynomial switching function
    double f = 1.0 - x * x * x * (6.0 * x * x - 15.0 * x + 10.0);
    double dfdr = -30.0 * w * x * x * (x - 1.0) * (x - 1.0);

    v = E * f;
    dvdr = dEdr * f + E * dfdr;
  } else {
    v = 0.0;
    dvdr = 0.0;
  }
}

/* ---------------------------------------------------------------------- */

void PairNLH::set_coeff(int i, int j, double zi, double zj, double a1_val, double a2_val,
                        double a3_val, double b1_val, double b2_val, double b3_val, double r1_val,
                        double r2_val)
{
  a1[i][j] = a1_val;
  a2[i][j] = a2_val;
  a3[i][j] = a3_val;
  b1[i][j] = b1_val;
  b2[i][j] = b2_val;
  b3[i][j] = b3_val;
  r1[i][j] = r1_val;
  r2[i][j] = r2_val;
  r1sq[i][j] = r1_val * r1_val;
  r2sq[i][j] = r2_val * r2_val;

  zze[i][j] = zi * zj * force->qqr2e * force->qelectron * force->qelectron;

  a1[j][i] = a1[i][j];
  a2[j][i] = a2[i][j];
  a3[j][i] = a3[i][j];
  b1[j][i] = b1[i][j];
  b2[j][i] = b2[i][j];
  b3[j][i] = b3[i][j];
  r1[j][i] = r1[i][j];
  r2[j][i] = r2[i][j];
  r1sq[j][i] = r1sq[i][j];
  r2sq[j][i] = r2sq[i][j];
  zze[j][i] = zze[i][j];
}