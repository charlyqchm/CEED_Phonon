#ifndef CEED_PHON_SUBSH
#define CEED_PHON_SUBSH

#include <iostream>
#include <vector>
#include <string>
#include <complex>
#include <time.h>
#include <fstream>
#include <stdlib.h>
#include <math.h>
#include <iomanip>
#include <limits>

using namespace std;
typedef unsigned int UNINT;

extern "C" { extern
   void dsyev_(char* jobz, char* uplo, int* n, double* a, int* lda,
               double* w, double* work, int* lwork, int* info);
}
extern "C" {
   void dgemm_(const char *TRANSA, const char *TRANSB, const int *M,
                      const int *N, const int *K, double *ALPHA, double *A,
                      const int *LDA, double *B, const int *LDB, double *BETA,
                      double *C, const int *LDC);
}


void read_inputs(UNINT& n_el, UNINT& n_phon, UNINT& np_levels, UNINT& n_tot,
                 UNINT& n_bath, vector<double>& el_ener_vec,
                 vector<double>& w_phon_vec, vector<double>& mass_phon_vec,
                 vector<double>& fb_vec);

void read_matrix_inputs(UNINT& n_el, UNINT& n_phon, UNINT& np_levels,
                        UNINT& n_tot, vector<double>& Fcoup_mat,
                        vector<double>& mu_elec_mat, vector<double>& dVdX_mat);

void init_matrix(vector < complex<double> >& H_tot, vector<double>& H0_mat,
                 vector<double>& Hcoup_mat, vector<double>& Fcoup_mat,
                 vector<double>& mu_elec_mat, vector<double>& mu_phon_mat,
                 vector<double>& v_bath_mat,
                 vector < complex<double> >& mu_tot,
                 vector<double>& dVdX_mat, vector<double>& ki_vec,
                 vector<double>& xi_vec, vector<double>& eigen_E,
                 vector<double>& eigen_coef, vector<double>& eigen_coefT,
                 vector < complex<double> >& rho_elec,
                 vector < complex<double> >& rho_phon,
                 vector < complex<double> >& rho_tot,
                 vector < complex<double> >& rho_new,
                 UNINT n_tot, UNINT n_el, UNINT n_phon, UNINT np_levels,
                 UNINT n_bath);

void build_rho_matrix(vector < complex<double> >& rho_tot,
                      vector<double>& eigen_coef, vector<double>& eigen_coefT,
                      UNINT n_tot);

void insert_eterm_in_bigmatrix(int ind_i, int ind_j , int n_el, int n_tot,
                               int n_phon, int np_levels, double Mij,
                               vector<double>& M_mat);

void build_matrix(vector < complex<double> >& H_tot, vector<double>& H0_mat,
                  vector<double>& Hcoup_mat, vector<double>& mu_phon_mat,
                  vector<double>& dVdX_mat, vector<double>& Fcoup_mat,
                  vector<double>& mu_elec_mat,
                  vector < complex<double> >& mu_tot,
                  vector<double>& el_ener_vec, vector<double>& w_phon_vec,
                  vector<double>& mass_phon_vec, double k0_inter, UNINT n_el,
                  UNINT n_phon, UNINT np_levels, UNINT n_tot);

void eigenval_elec_calc(vector<double>& mat, vector<double>& eigenval,
                        vector<double>& coef, UNINT ntotal);

void matmul_blas(vector<double>& matA, vector<double>& matB,
                 vector<double>& matC, int ntotal);
#endif
