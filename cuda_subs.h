#ifndef CUDA_SUBS
#define CUDA_SUBS

#include <cuda.h>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <ctime>
#include <cstdlib>
#include <complex>

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
extern cuDoubleComplex  *dev_dvdx;
extern double           *dev_vbath;
extern double           *dev_fb;
extern double           *dev_xi;
extern double           *dev_vi;
extern double           *dev_ki;
extern double           *dev_xf;
extern double           *dev_vf;
extern double           *dev_xh;
extern UNINT             Ncores1;
extern UNINT             Ncores2;
const  UNINT             Nthreads = 512;

void init_cuda(complex<double> *H_tot, complex<double> *mu_tot,
               double *v_bath_mat, double *fb_vec, double *xi_vec,
               double *vi_vec, double *ki_vec,
               complex<double> *rho_tot,
               complex<double> *rho_phon, complex<double> *dVdX_mat,
               UNINT n_el, UNINT n_phon, UNINT np_levels, UNINT n_tot,
               UNINT n_bath);

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

void runge_kutta_propagator_cuda(double a_ceed, double dt, double Efield,
                                 double *fb_vec, int tt,
                                 UNINT n_el, UNINT n_phon, UNINT np_levels,
                                 UNINT n_tot, UNINT n_bath);

void calcrhophon(cuDoubleComplex *dev_rhoin, int n_el, int n_phon,
                 int np_levels, int n_tot);
void getingmat(complex<double> *matA, cuDoubleComplex *dev_A, int n_tot);
#endif
