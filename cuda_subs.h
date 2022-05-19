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
extern cuDoubleComplex  *dev_Xmat;
extern cuDoubleComplex  *dev_Pmat;
extern UNINT             Ncores1;
extern UNINT             Ncores2;
const  UNINT             Nthreads = 512;

void init_cuda(complex<double> *H_tot, complex<double> *mu_tot,
               complex<double> *rho_tot,
               complex<double> *rho_phon,
               complex<double> *X_phon_mat,
               complex<double> *P_phon_mat,
               UNINT n_el, UNINT n_phon, UNINT np_levels, UNINT n_tot);

void free_cuda_memory();

void matmul_cublas(cuDoubleComplex *dev_A, cuDoubleComplex *dev_B,
                   cuDoubleComplex *dev_C, int dim);

void commute_cuda(cuDoubleComplex *dev_A, cuDoubleComplex *dev_B,
                  cuDoubleComplex *dev_C, int dim, const cuDoubleComplex alf);

void anticommute_cuda(cuDoubleComplex *dev_A, cuDoubleComplex *dev_B,
                      cuDoubleComplex *dev_C, int dim,
                      const cuDoubleComplex alf);

void matadd_cublas(cuDoubleComplex *dev_A, cuDoubleComplex *dev_B,
                   cuDoubleComplex *dev_C, int dim, const cuDoubleComplex alf,
                   const cuDoubleComplex bet);

double get_trace_cuda(cuDoubleComplex *dev_A, UNINT dim);

void include_Hceed_cuda(cuDoubleComplex *dev_Hout, cuDoubleComplex *dev_Hin,
                        cuDoubleComplex *dev_mu, cuDoubleComplex *dev_rhoin,
                        double a_ceed, int n_tot);

double get_Qforces_cuda(cuDoubleComplex *dev_rhoin ,double *fb_vec,
                        UNINT n_el, UNINT n_phon, UNINT np_levels, UNINT n_tot);

void runge_kutta_propagator_cuda(double mass_bath, double a_ceed, double dt,
                                 double Efield, double Efieldaux,
                                 double C_term, double LM_term,
                                 int tt, UNINT n_el,
                                 UNINT n_phon, UNINT np_levels,
                                 UNINT n_tot);

void calcrhophon(cuDoubleComplex *dev_rhoin, int n_el, int n_phon,
                 int np_levels, int n_tot);
void getingmat(complex<double> *matA, cuDoubleComplex *dev_A, int n_tot);

void getting_printing_info(double *Ener, double *mu, complex<double> *tr_rho,
                           complex<double> *rho_tot, UNINT n_tot);

#endif
