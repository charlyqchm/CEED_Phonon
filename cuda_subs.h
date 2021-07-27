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

extern cuDoubleComplex  *dev_rhoelec;
extern cuDoubleComplex  *dev_rhophon;
extern cuDoubleComplex  *dev_rhotot;
extern cuDoubleComplex  *dev_rhonew;
extern cuDoubleComplex  *dev_Htot1;
extern cuDoubleComplex  *dev_Htot2;
extern cuDoubleComplex  *dev_Htot3;
extern cuDoubleComplex  *dev_mutot;
extern double           *dev_vbath;
extern double           *dev_fb;
extern UNINT             Ncores;
const  UNINT             Nthreads = 512;

void init_cuda(complex<double> *H_tot, complex<double> *mu_tot,
               double *v_bath_mat, double *fb_vec, complex<double> *rho_tot,
               UNINT n_el, UNINT n_phon, UNINT n_tot);

void free_cuda_memory();

void matmul_cublas(cuDoubleComplex *dev_A, cuDoubleComplex *dev_B,
                   cuDoubleComplex *dev_C, int dim);

void commute_cuda(cuDoubleComplex *dev_A, cuDoubleComplex *dev_B,
                  cuDoubleComplex *dev_C, int dim);

double get_trace_cuda(cuDoubleComplex *dev_A, UNINT dim);

void include_Hceed_cuda(cuDoubleComplex *dev_Hout, cuDoubleComplex *dev_Hin,
                        cuDoubleComplex *dev_Hceed,
                        cuDoubleComplex *dev_mu, cuDoubleComplex *dev_rhoaux,
                        double a_ceed, int n_tot);


void getingmat(complex<double> *matA, cuDoubleComplex *dev_A, int n_tot);
#endif
