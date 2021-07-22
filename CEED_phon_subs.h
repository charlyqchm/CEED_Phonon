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
//#include <cuda.h>
//#include <cuComplex.h>
//#include <ctime>
//#include <cstdlib>

using namespace std;
typedef unsigned int UNINT;

void init_matrix(vector<double>& H0_mat, vector<double>& Hcoup_mat,
                 vector<double>& Fcoup_mat, vector<double>& mu_elec_mat,
                 vector<double>& mu_phon_mat, vector<double>& v_bath_mat,
                 UNINT n_tot);
void read_inputs(UNINT& n_el, UNINT& n_phon, UNINT& np_levels, UNINT& n_tot,
                 vector<double>& H0_mat, vector<double>& Hcoup_mat,
                 vector<double>& mu_elec_mat, vector<double>& Fcoup_mat,
                 vector<double>& mu_phon_mat, vector<double>& v_bath_mat,
                 vector<double>& el_ener_vec, vector<double>& w_phon_vec,
                 vector<double>& mass_phon_vec);
void insert_eterm_in_bigmatrix(int ind_i, int ind_j , int n_el, int n_tot,
                               int n_phon, int np_levels, double Mij,
                               vector<double>& M_mat);
void build_matrix(vector<double>& H0_mat, vector<double>& Hcoup_mat,
                  vector<double>& mu_phon_mat,vector<double>& el_ener_vec,
                  vector<double>& w_phon_vec, vector<double>& mass_phon_vec,
                  UNINT n_el, UNINT n_phon, UNINT np_levels, UNINT n_tot);

#endif
