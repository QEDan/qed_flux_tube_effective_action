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
    
    double r_eps = r + 1e-15;
    double v_ml = e * s3 * (a/r_eps + da) + (ml*ml - 1.0)/(r_eps*r_eps) + e*e*a*a - 2.0*e*ml*a/r_eps;
    
    // Convert chi to complex
    double complex chi = params.chi.real + params.chi.imag * I;
    
    // ODE: u'' + 1/r * u' - (v_ml + 1/r^2 - (chi^2 - m^2)) u = 0
    // Effective potential term for u''
    double complex v_eff = v_ml + 1.0/(r_eps*r_eps) - (chi * chi - params.m * params.m);
    
    ds.du = -1.0/r_eps * s.du + v_eff * s.u;
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
    
    double complex chi = params.chi.real + params.chi.imag * I;
    double m = params.m;
    double e = params.e;
    
    // 1. Forward Integration for u0
    double rho_min = profile.rho[0];
    State s0;
    int abs_ml = abs(params.ml);
    
    // Improved IC (Second order)
    double a0 = profile.a_phi[0];
    double da0 = profile.da_phi[0];
    double s3 = (double)params.sigma3;
    double v_ml0 = e * s3 * (a0/rho_min + da0) + (abs_ml*abs_ml - 1.0)/(rho_min*rho_min) + e*e*a0*a0 - 2.0*e*abs_ml*a0/rho_min;
    double complex v_eff0 = v_ml0 + 1.0/(rho_min*rho_min) - (chi*chi - m*m);
    double complex k_eff2 = -(v_eff0 - (double)(abs_ml*abs_ml)/(rho_min*rho_min));

    if (abs_ml == 0) {
        s0.u = 1.0 - k_eff2 * rho_min * rho_min / 4.0;
        s0.du = -k_eff2 * rho_min / 2.0;
    } else {
        double complex scale = 1.0 - k_eff2 * rho_min * rho_min / (4.0 * (abs_ml + 1.0));
        s0.u = cpow(rho_min, abs_ml) * scale;
        s0.du = abs_ml * cpow(rho_min, abs_ml - 1.0) * scale - cpow(rho_min, abs_ml) * (k_eff2 * rho_min / (2.0 * (abs_ml + 1.0)));
    }
    
    u0[0] = s0.u;
    State curr = s0;
    for (int i = 0; i < N - 1; i++) {
        double h = profile.rho[i+1] - profile.rho[i];
        double a_mid = 0.5 * (profile.a_phi[i] + profile.a_phi[i+1]);
        double da_mid = 0.5 * (profile.da_phi[i] + profile.da_phi[i+1]);
        curr = rk4_step(profile.rho[i], h, curr, params, 
                        profile.a_phi[i], profile.da_phi[i],
                        a_mid, da_mid,
                        profile.a_phi[i+1], profile.da_phi[i+1]);
        u0[i+1] = curr.u;

        // Delta jump condition
        if (params.lambd > 0 && profile.rho[i] < params.lambd && profile.rho[i+1] >= params.lambd) {
            double F_cal = params.e * params.F / (2.0 * M_PI);
            double jump_coeff = -2.0 * F_cal / (params.lambd * params.lambd);
            curr.du += jump_coeff * curr.u;
        }
    }
    double complex du0_last = curr.du;

    // 2. Backward Integration for uinf
    double rho_max = profile.rho[N-1];
    double complex k2_ext = chi * chi - params.m * params.m;
    if (cabs(k2_ext) < 1e-12) k2_ext = 1e-12;
    double complex k_ext = csqrt(k2_ext);
    
    // Ensure Im(k) >= 0 for stability/decay
    if (cimag(k_ext) < 0) k_ext = -k_ext;

    double F_cal_ext = params.e * params.F / (2.0 * M_PI);
    double n_order = (double)params.ml - F_cal_ext;

    State sinf;
    // For consistency with PyTorch yv/kv switching:
    if (creal(k2_ext) > 0 && fabs(cimag(k2_ext)) < 1e-10) {
        // Oscillatory (Real k)
        double k_val = creal(k_ext);
        double phase = k_val * rho_max - n_order * M_PI / 2.0 - M_PI / 4.0;
        double ampl = sqrt(2.0 / (M_PI * k_val * rho_max));
        sinf.u = ampl * sin(phase);
        sinf.du = k_val * ampl * cos(phase) - 0.5 * sinf.u / rho_max;
    } else {
        // Complex or Decaying
        double complex kappa = -I * k_ext;
        if (creal(kappa) < 0) kappa = -kappa;
        double complex c_ampl = csqrt(M_PI / (2.0 * kappa * rho_max));
        sinf.u = c_ampl * cexp(-kappa * rho_max);
        sinf.du = (-kappa - 0.5/rho_max) * sinf.u;
    }
    
    uinf[N-1] = sinf.u;
    curr = sinf;
    for (int i = N - 1; i > 0; i--) {
        double h = profile.rho[i-1] - profile.rho[i];

        double a_mid = 0.5 * (profile.a_phi[i] + profile.a_phi[i-1]);
        double da_mid = 0.5 * (profile.da_phi[i] + profile.da_phi[i-1]);
        curr = rk4_step(profile.rho[i], h, curr, params, 
                        profile.a_phi[i], profile.da_phi[i],
                        a_mid, da_mid,
                        profile.a_phi[i-1], profile.da_phi[i-1]);
        uinf[i-1] = curr.u;

        // Delta jump condition (backward)
        if (params.lambd > 0 && profile.rho[i] > params.lambd && profile.rho[i-1] <= params.lambd) {
            double F_cal = params.e * params.F / (2.0 * M_PI);
            double jump_coeff = 2.0 * F_cal / (params.lambd * params.lambd);
            curr.du += jump_coeff * curr.u;
        }
    }

    // Compute Wronskian W0 = rho * (u0' * uinf - u0 * uinf')
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
