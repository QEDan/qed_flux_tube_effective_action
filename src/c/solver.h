#ifndef SOLVER_H
#define SOLVER_H

#include <complex.h>

typedef struct {
    double real;
    double imag;
} Complex128;

typedef struct {
    Complex128 chi;
    int ml;
    int sigma3;
    double m;
    double e;
    double lambd;
    double F;
} Parameters;

typedef struct {
    double* rho;
    double* a_phi;
    double* da_phi;
    int n_points;
} Profile;

// Solves the radial ODE and returns the value of [u0(rho)*uinf(rho)/W0] at each rho point.
// results must be an allocated array of size profile.n_points
void solve_greens_function(Parameters params, Profile profile, double complex* results);

// Batch solver using OpenMP
void solve_batch(Parameters* params_array, int n_params, Profile profile, double complex* results_array);

#endif
