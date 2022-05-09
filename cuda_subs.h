#ifndef CUDA_SUBS
#define CUDA_SUBS

#include <math.h>
#include <cuda.h>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <ctime>
#include <cstdlib>
#include <complex>
#include <stdio.h>

using namespace std;
typedef unsigned int UNINT;

extern cuDoubleComplex  *dev_rhophon;
extern cuDoubleComplex  *dev_rhotot;
extern cuDoubleComplex  *dev_rhonew;
extern cuDoubleComplex  *dev_rhoaux;
extern cuDoubleComplex  *dev_Drho;
extern cuDoubleComplex  *dev_Htot1;
extern cuDoubleComplex  *dev_Htot2;
extern cuDoubleComplex  *dev_Htot3;
extern cuDoubleComplex  *dev_mutot;
extern double           *dev_etaL_ke;
extern double           *dev_lambdaL_ke;
extern double           *dev_etaS_ke;
extern double           *dev_lambdaS_ke;
extern double           *dev_ke_del1;
extern double           *dev_ke_del2;
extern double           *dev_Nphon_ke;
extern int              *dev_keind_j;
extern int              *dev_keind_k;
extern UNINT             Ncores1;
extern UNINT             Ncores2;
extern UNINT             Ncores3;
const  UNINT             Nthreads = 512;

void init_cuda(complex<double> *H_tot, complex<double> *mu_tot,
               complex<double> *rho_tot,
               complex<double> *rho_phon,
               int *ke_index_i, int *ke_index_j, int *ke_index_k,
               double *ke_delta1_vec, double *ke_delta2_vec,
               double *ke_N_phon_vec,
               UNINT n_el, UNINT n_phon, UNINT np_levels, UNINT n_tot,
               UNINT n_ke_bath, UNINT n_ke_inter);

void free_cuda_memory();

void matmul_cublas(cuDoubleComplex *dev_A, cuDoubleComplex *dev_B,
                   cuDoubleComplex *dev_C, int dim);

void commute_cuda(cuDoubleComplex *dev_A, cuDoubleComplex *dev_B,
                  cuDoubleComplex *dev_C, int dim, const cuDoubleComplex alf);

void matadd_cublas(cuDoubleComplex *dev_A, cuDoubleComplex *dev_B,
                   cuDoubleComplex *dev_C, int dim, const cuDoubleComplex alf,
                   const cuDoubleComplex bet);

double get_trace_cuda(cuDoubleComplex *dev_A, UNINT dim);

void include_Hceed_cuda(cuDoubleComplex *dev_Hout, cuDoubleComplex *dev_Hin,
                        cuDoubleComplex *dev_mu, cuDoubleComplex *dev_rhoin,
                        double a_ceed, int n_tot);

double get_Qforces_cuda(cuDoubleComplex *dev_rhoin ,double *fb_vec,
                        UNINT n_el, UNINT n_phon, UNINT np_levels, UNINT n_tot);

void include_ke_terms(cuDoubleComplex *dev_rho, cuDoubleComplex *dev_Drho,
                      int *ke_index_i, double *eta_s_vec,
                      double *lambda_s_vec, double *eta_l_vec,
                      double *lambda_l_vec, UNINT n_tot, UNINT n_ke_inter);

void runge_kutta_propagator_cuda(double mass_bath, double a_ceed, double dt,
                                 double Efield, double Efieldaux,
                                 double *eta_s_vec,
                                 double *lambda_s_vec, double *eta_l_vec,
                                 double *lambda_l_vec, int *ke_index_i,
                                 int tt, UNINT n_el,
                                 UNINT n_phon, UNINT np_levels,
                                 UNINT n_tot, UNINT n_ke_inter);

void calcrhophon(cuDoubleComplex *dev_rhoin, int n_el, int n_phon,
                 int np_levels, int n_tot);
void getingmat(complex<double> *matA, cuDoubleComplex *dev_A, int n_tot);

void getting_printing_info(double *Ener, double *mu, complex<double> *tr_rho,
                           complex<double> *rho_tot, UNINT n_tot);

#endif
