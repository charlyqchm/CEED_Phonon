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

void read_matrix_inputs(UNINT& n_el, UNINT& n_phon, UNINT& np_levels,
                        UNINT& n_tot, vector<double>& Fcoup_mat,
                        vector<double>& mu_elec_mat, vector<double>& Fbath_mat,
                        vector<double>& w_phon_vec,
                        vector<double>& mass_phon_vec);

void init_matrix(vector < complex<double> >& H_tot, vector<double>& H0_mat,
                 vector<double>& Hcoup_mat, vector<double>& Fcoup_mat,
                 vector<double>& mu_elec_mat, vector<double>& mu_phon_mat,
                 vector < complex<double> >& mu_tot,
                 vector<double>& Efield_t,
                 vector<double>& eigen_E,
                 vector<double>& eigen_coef, vector<double>& eigen_coefT,
                 vector < complex<double> >& rho_phon,
                 vector < complex<double> >& rho_tot,
                 UNINT n_tot, UNINT n_el, UNINT n_phon, UNINT np_levels,
                 UNINT n_ke_bath, vector<double>& w_ke_vec,
                 vector<double>& eta_s_vec, vector<double>& lambda_s_vec,
                 vector<double>& Fbath_mat);

void build_rho_matrix(vector < complex<double> >& rho_tot,
                      vector<double>& eigen_coef, vector<double>& eigen_coefT,
                      vector<double>& mass_phon_vec, vector<double>& w_phon_vec,
                      UNINT n_tot, UNINT n_el, UNINT n_phon, UNINT np_levels);

void insert_eterm_in_bigmatrix(int ind_i, int ind_j , int n_el, int n_tot,
                               int n_phon, int np_levels, double Mij,
                               vector<double>& M_mat);

void build_matrix(vector < complex<double> >& H_tot, vector<double>& H0_mat,
                  vector<double>& Hcoup_mat, vector<double>& mu_phon_mat,
                  vector<double>& Fcoup_mat,
                  vector<double>& mu_elec_mat,
                  vector < complex<double> >& mu_tot,
                  vector<double>& el_ener_vec, vector<double>& w_phon_vec,
                  vector<double>& mass_phon_vec, UNINT n_el,
                  UNINT n_phon, UNINT np_levels, UNINT n_tot);

void eigenval_elec_calc(vector<double>& mat, vector<double>& eigenval,
                        vector<double>& coef, UNINT ntotal);

void matmul_blas(vector<double>& matA, vector<double>& matB,
                 vector<double>& matC, int ntotal);

double rand_normal();

double rand_gaussian(double mean, double stdev);

void init_bath(double temp, double bmass, double ki, double span, int seed,
               UNINT n_ke_bath, vector<double>& w_ke_vec);

void init_output(ofstream* outfile);

void write_output(double mass_bath, double dt, int tt, int print_t, UNINT n_el,
                  UNINT n_phon, UNINT np_levels, UNINT n_tot,
                  vector < complex<double> >& rho_tot, ofstream* outfile);

void readinput(UNINT& n_el, UNINT& n_phon, UNINT& np_levels, UNINT& n_tot,
               int& t_steps, int& print_t, double& dt,
               double& Efield, double& b_temp,
               double& a_ceed, int& seed, UNINT& n_ke_bath, double& sigma_ke,
               double& k0_ke_inter, vector<double>& el_ener_vec,
               vector<double>& w_phon_vec, vector<double>& mass_phon_vec);

void readefield(int& efield_flag, vector<double>& efield_vec);
void efield_t(int efield_flag, int tt, double dt, double Efield,
                 vector<double> efield_vec, vector<double>& Efield_t);

void getting_ke_terms(UNINT n_tot, UNINT n_ke_bath, UNINT& n_ke_inter,
                      UNINT n_el, UNINT n_phon, UNINT& np_levels,
                      double mass_bath, vector<int>& ke_index_i,
                      vector<int>& ke_index_j, vector<int>& ke_index_k,
                      double sigma_ke, double k0_ke_inter,
                      vector<double>& eigen_E, vector<double>& w_ke_vec,
                      vector<double>& ke_delta1_vec,
                      vector<double>& ke_delta2_vec,
                      vector<double>& eta_l_vec,
                      vector<double>& lambda_l_vec,
                      vector<double>& w_phon_vec,
                      vector<double>& mass_phon_vec,
                      vector<double>& Fbath_mat);

void creating_ke_bath(UNINT n_ke_bath, vector<double>& w_ke_vec,
                      vector<double>& ke_N_phon_vec, double b_temp);

double dirac_delta(double Ea, double Eb, double wj, double sigma);

double factorial(int n);

double franck_condon_term(double alpha, int ind_i, int ind_j);

#endif
