#include "CEED_phon_subs.h"


//##############################################################################
void init_matrix(vector<double>& H0_mat, vector<double>& Hcoup_mat,
                 vector<double>& Fcoup_mat, vector<double>& mu_elec_mat,
                 vector<double>& mu_phon_mat, vector<double>& v_bath_mat,
                 UNINT n_tot){

   for(int ii=0; ii<(n_tot*n_tot); ii++){
      H0_mat.push_back(0.0e0);
      Hcoup_mat.push_back(0.0e0);
      Fcoup_mat.push_back(0.0e0);
      mu_elec_mat.push_back(0.0e0);
      mu_phon_mat.push_back(0.0e0);
      v_bath_mat.push_back(0.0e0);
   }

   return;
}
//##############################################################################
void read_inputs(UNINT& n_el, UNINT& n_phon, UNINT& np_levels, UNINT& n_tot,
                 vector<double>& H0_mat, vector<double>& Hcoup_mat,
                 vector<double>& mu_elec_mat, vector<double>& Fcoup_mat,
                 vector<double>& mu_phon_mat, vector<double>& v_bath_mat,
                 vector<double>& el_ener_vec, vector<double>& w_phon_vec,
                 vector<double>& mass_phon_vec){

   int      n_inter; //number of interactions (dipole or phonon coupling)
   ifstream inputf;
   int      ind_i;
   int      ind_j;
   double   Mij;
   ofstream output_test;


// Reading Basic variables:
   inputf.open("input.in");
   if (!inputf) {
      cout << "Unable to open input.in";
      exit(1); // terminate with error
   }

   inputf>> n_el;
   inputf>> n_phon;
   inputf>> np_levels;
   n_tot = n_el * n_phon * np_levels;

   for (int ii=0; ii<n_el; ii++){
      double el_ener;
      inputf>> el_ener;
      el_ener_vec.push_back(el_ener);
   }

   for (int ii=0; ii<n_phon; ii++){
      double w_phon;
      inputf>> w_phon;
      w_phon_vec.push_back(w_phon);
   }

   for (int ii=0; ii<n_phon; ii++){
      double mass_phon;
      inputf>> mass_phon;
      mass_phon_vec.push_back(mass_phon);
   }

   inputf.close();

   init_matrix(H0_mat, Hcoup_mat, Fcoup_mat, mu_elec_mat, mu_phon_mat,
               v_bath_mat, n_tot);

   inputf.open("Fmat.in");
   if (!inputf) {
      cout << "Unable to open Fmat.in";
      exit(1); // terminate with error
   }

   inputf>> n_inter;
   for (int ii=0; ii<n_inter; ii++){
      inputf>> ind_i;
      inputf>> ind_j;
      inputf>> Mij;
      insert_eterm_in_bigmatrix(ind_i, ind_j, n_el, n_tot, n_phon, np_levels,
                                Mij, Fcoup_mat);
   }
   inputf.close();

   inputf.open("mumat.in");
   if (!inputf) {
      cout << "Unable to open mumat.in";
      exit(1); // terminate with error
   }

   inputf>> n_inter;
   for (int ii=0; ii<n_inter; ii++){
      inputf>> ind_i;
      inputf>> ind_j;
      inputf>> Mij;
      insert_eterm_in_bigmatrix(ind_i, ind_j, n_el, n_tot, n_phon, np_levels,
                                Mij, mu_elec_mat);
   }
   inputf.close();

   return;

}
//##############################################################################
void insert_eterm_in_bigmatrix(int ind_i, int ind_j , int n_el, int n_tot,
                               int n_phon, int np_levels, double Mij,
                               vector<double>& M_mat){

   for (int ii=0; ii<n_phon; ii++){
   for (int jj=0; jj<np_levels; jj++){
      int ind1 = jj + ii*n_phon + ind_i*n_phon*np_levels;
      int ind2 = jj + ii*n_phon + ind_j*n_phon*np_levels;
      int ind3 = ind1 + (ind2 * n_tot);
      int ind4 = ind2 + (ind1 * n_tot);

      M_mat[ind3] = Mij;
      M_mat[ind4] = M_mat[ind3];

   }
   }

   return;
}
//##############################################################################
void build_matrix(vector<double>& H0_mat, vector<double>& Hcoup_mat,
                  vector<double>& mu_phon_mat,vector<double>& el_ener_vec,
                  vector<double>& w_phon_vec, vector<double>& mass_phon_vec,
                  UNINT n_el, UNINT n_phon, UNINT np_levels, UNINT n_tot){

   for (int ii=0; ii<n_el; ii++){
   for (int jj=0; jj<n_phon; jj++){
      double mwj = w_phon_vec[jj] * mass_phon_vec[jj];
      for (int kk=0; kk<np_levels; kk++){
         int ind1 = kk + jj*n_phon + ii*n_phon*np_levels;
         int ind2 = ind1 + (ind1 * n_tot);
         H0_mat[ind2] = el_ener_vec[ii] + w_phon_vec[jj]*(0.5e0 + kk);
      }
      for (int kk=0; kk<np_levels-1; kk++){
         int ind1   = kk + jj*n_phon + ii*n_phon*np_levels;
         int ind2   = ind1 + ((ind1+1) * n_tot);
         int ind3   = ind1+1 + (ind1 * n_tot);
         double Mij = sqrt((kk+1.0)/(2.0*mwj));
         mu_phon_mat[ind2] += Mij;
         mu_phon_mat[ind3] += Mij;

         if (kk > 0){
            ind2 = ind1-1 + (ind1 * n_tot);
            ind3 = ind1 + ((ind1-1) * n_tot);
            Mij  = sqrt(kk/(2.0*mwj));
            mu_phon_mat[ind2] += Mij;
            mu_phon_mat[ind3] += Mij;
         }
      }
   }
   }

   return;

}
//##############################################################################

//##############################################################################
