#include "CEED_phon_subs.h"

int main(){

   UNINT             n_el;
   UNINT             n_phon;
   UNINT             np_levels;
   UNINT             n_tot;
   vector<double>    el_ener_vec;
   vector<double>    w_phon_vec;
   vector<double>    mass_phon_vec;
   vector<double>    Fcoup_mat;
   vector<double>    mu_elec_mat;
   vector<double>    mu_phon_mat;
   vector<double>    v_bath_mat;
   vector<double>    H0_mat;
   vector<double>    Hcoup_mat;

   ofstream output_test;

   read_inputs(n_el, n_phon, np_levels, n_tot, H0_mat, Hcoup_mat, mu_elec_mat,
               Fcoup_mat, mu_phon_mat, v_bath_mat, el_ener_vec, w_phon_vec,
               mass_phon_vec);

   build_matrix(H0_mat, Hcoup_mat, mu_phon_mat, el_ener_vec, w_phon_vec,
                mass_phon_vec, n_el,n_phon, np_levels, n_tot);

   // output_test.open("file.out");
   // for (int jj=0; jj<n_tot; jj++){
   // for (int ii=0; ii<n_tot; ii++){
   //    output_test<<ii<<"  "<<jj<<"  "<<Fcoup_mat[ii+jj*n_tot]<<endl;
   // }
   // }
   //
   // output_test.close();

};
