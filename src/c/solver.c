#include "solver.h"
#include <math.h>
#include <stdlib.h>
#include <complex.h>
#include <omp.h>

typedef struct {
    double complex u;
    double complex du;
} State;

static State f(double r, State s, Parameters params, double a, double da) {
    State ds;
    ds.u = s.du;
    
    double e = params.e;
    double ml = (double)params.ml;
    double s3 = (double)params.sigma3;
    
    double v_ml = e * s3 * (a/r + da) + (ml*ml - 1.0)/(r*r) + e*e*a*a - 2.0*e*ml*a/r;
    
    // ODE: u'' + 1/r * u' - (v_ml + 1/r^2 - (chi^2 - m^2)) u = 0
    // Effective potential term for u''
    double complex v_eff = v_ml + 1.0/(r*r) - (params.chi * params.chi - params.m * params.m);
    
    ds.du = -1.0/r * s.du + v_eff * s.u;
    return ds;
}

static State rk4_step(double r, double h, State s, Parameters params, 
                      double a_start, double da_start, 
                      double a_mid, double da_mid, 
                      double a_end, double da_end) {
    State k1 = f(r, s, params, a_start, da_start);
    
    State s2 = {s.u + 0.5 * h * k1.u, s.du + 0.5 * h * k1.du};
    State k2 = f(r + 0.5 * h, s2, params, a_mid, da_mid);
    
    State s3 = {s.u + 0.5 * h * k2.u, s.du + 0.5 * h * k2.du};
    State k3 = f(r + 0.5 * h, s3, params, a_mid, da_mid);
    
    State s4 = {s.u + h * k3.u, s.du + h * k3.du};
    State k4 = f(r + h, s4, params, a_end, da_end);
    
    State res;
    res.u = s.u + (h/6.0) * (k1.u + 2.0*k2.u + 2.0*k3.u + k4.u);
    res.du = s.du + (h/6.0) * (k1.du + 2.0*k2.du + 2.0*k3.du + k4.du);
    return res;
}

void solve_greens_function(Parameters params, Profile profile, double complex* results) {
    int N = profile.n_points;
    double complex* u0 = malloc(N * sizeof(double complex));
    double complex* uinf = malloc(N * sizeof(double complex));
    
    // Boundary condition for u0 at small rho
    double rho_min = profile.rho[0];
    State s0;
    int abs_ml = abs(params.ml);
    if (abs_ml == 0) {
        s0.u = 1.0;
        s0.du = 0.0;
    } else {
        s0.u = cpow(rho_min, abs_ml);
        s0.du = abs_ml * cpow(rho_min, abs_ml - 1);
    }
    
    u0[0] = s0.u;
    State curr = s0;
    for (int i = 0; i < N - 1; i++) {
        double h = profile.rho[i+1] - profile.rho[i];
        
        // Delta jump condition
        if (params.lambd > 0 && profile.rho[i] < params.lambd && profile.rho[i+1] >= params.lambd) {
            double F_cal = params.e * params.F / (2.0 * M_PI);
            double jump_coeff = -2.0 * F_cal / (params.lambd * params.lambd);
            curr.du += jump_coeff * curr.u;
        }
        
        double a_mid = 0.5 * (profile.a_phi[i] + profile.a_phi[i+1]);
        double da_mid = 0.5 * (profile.da_phi[i] + profile.da_phi[i+1]);
        curr = rk4_step(profile.rho[i], h, curr, params, 
                        profile.a_phi[i], profile.da_phi[i],
                        a_mid, da_mid,
                        profile.a_phi[i+1], profile.da_phi[i+1]);
        u0[i+1] = curr.u;
    }
    double complex du0_last = curr.du;

    // Boundary condition for uinf at large rho
    double rho_max = profile.rho[N-1];
    double complex k = csqrt(params.chi * params.chi - params.m * params.m);
    // Ensure Re(k) > 0 for decay or Im(k) > 0
    if (creal(k) < 0) k = -k;
    
    State sinf;
    sinf.u = cexp(-k * rho_max) / csqrt(rho_max);
    sinf.du = (-k - 0.5/rho_max) * sinf.u;
    
    uinf[N-1] = sinf.u;
    curr = sinf;
    for (int i = N - 1; i > 0; i--) {
        double h = profile.rho[i-1] - profile.rho[i];

        // Delta jump condition (backward)
        if (params.lambd > 0 && profile.rho[i] > params.lambd && profile.rho[i-1] <= params.lambd) {
            double F_cal = params.e * params.F / (2.0 * M_PI);
            double jump_coeff = 2.0 * F_cal / (params.lambd * params.lambd);
            curr.du += jump_coeff * curr.u;
        }

        double a_mid = 0.5 * (profile.a_phi[i] + profile.a_phi[i-1]);
        double da_mid = 0.5 * (profile.da_phi[i] + profile.da_phi[i-1]);
        curr = rk4_step(profile.rho[i], h, curr, params, 
                        profile.a_phi[i], profile.da_phi[i],
                        a_mid, da_mid,
                        profile.a_phi[i-1], profile.da_phi[i-1]);
        uinf[i-1] = curr.u;
    }

    // Compute Wronskian W0 = rho * (u0' * uinf - u0 * uinf')
    // We can compute it at rho_max
    double complex W0 = rho_max * (du0_last * uinf[N-1] - u0[N-1] * sinf.du);
    
    for (int i = 0; i < N; i++) {
        results[i] = (profile.rho[i] * u0[i] * uinf[i]) / W0;
    }
    
    free(u0);
    free(uinf);
}

void solve_batch(Parameters* params_array, int n_params, Profile profile, double complex* results_array) {
    #pragma omp parallel for
    for (int i = 0; i < n_params; i++) {
        solve_greens_function(params_array[i], profile, &results_array[i * profile.n_points]);
    }
}
