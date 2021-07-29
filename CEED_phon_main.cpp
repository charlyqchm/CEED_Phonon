#include "cuda_subs.h"
#include "CEED_phon_subs.h"

int main(){

   UNINT                      n_el;
   UNINT                      n_phon;
   UNINT                      np_levels;
   UNINT                      n_tot;
   UNINT                      n_bath;
   int                        t_steps;
   double                     dt;
   double                     k0_inter = 0.0e0; //interaction constant bath-phonons
   vector<double>             el_ener_vec;
   vector<double>             fb_vec;
   vector<double>             ki_vec;      //bath string constants
   vector<double>             xi_vec;      //bath coordinates
   vector<double>             vi_vec;      //bath coordinates
   vector<double>             w_phon_vec;
   vector<double>             mass_phon_vec;
   vector<double>             Fcoup_mat;
   vector<double>             mu_elec_mat;
   vector<double>             mu_phon_mat;
   vector<double>             v_bath_mat;
   vector<double>             H0_mat;
   vector<double>             Hcoup_mat;
   vector<double>             eigen_E;
   vector<double>             eigen_coef;
   vector<double>             eigen_coefT;
   vector < complex<double> > H_tot;
   vector < complex<double> > dVdX_mat;
   vector < complex<double> > mu_tot;
   vector < complex<double> > rho_phon;
   vector < complex<double> > rho_tot;

   ofstream output_test;

   read_inputs(n_el, n_phon, np_levels, n_tot, n_bath, el_ener_vec, w_phon_vec,
               mass_phon_vec, fb_vec);

   init_matrix(H_tot, H0_mat, Hcoup_mat, Fcoup_mat, mu_elec_mat, mu_phon_mat,
               v_bath_mat, mu_tot, dVdX_mat, ki_vec, xi_vec, vi_vec, eigen_E,
               eigen_coef, eigen_coefT, rho_phon, rho_tot,
               n_tot, n_el, n_phon, np_levels, n_bath);

   read_matrix_inputs(n_el, n_phon, np_levels, n_tot, Fcoup_mat, mu_elec_mat);

   build_matrix(H_tot, H0_mat, Hcoup_mat, mu_phon_mat, dVdX_mat, Fcoup_mat,
                mu_elec_mat, mu_tot, el_ener_vec, w_phon_vec, mass_phon_vec,
                k0_inter, n_el, n_phon, np_levels, n_tot);

   eigenval_elec_calc(H0_mat, eigen_E, eigen_coef, n_tot);
   for(int jj=0; jj < n_tot; jj++){
   for(int ii=0; ii < n_tot; ii++){
      eigen_coefT[jj + ii * n_tot] = eigen_coef[ii + jj * n_tot];
   }
   }

   build_rho_matrix(rho_tot, eigen_coef, eigen_coefT, n_tot);

   init_cuda(& *H_tot.begin(), & *mu_tot.begin(), & *v_bath_mat.begin(),
             & *fb_vec.begin(), & *xi_vec.begin(), & *vi_vec.begin(),
             & *ki_vec.begin(), & *rho_tot.begin(), & *rho_phon.begin(),
             & *dVdX_mat.begin(), n_el, n_phon, np_levels, n_tot, n_bath);

//TESTING CUDA
   // commute_cuda(dev_Htot1, dev_rhotot, dev_rhonew, n_tot);

   getingmat(& *H_tot.begin(), dev_rhonew, n_tot);

   free_cuda_memory();
   output_test.open("file.out");
   for(int jj=0; jj<n_tot; jj++){
      for(int ii=0; ii<n_tot; ii++){
         output_test<<ii<<"  "<<jj<<"  "<<H_tot[ii+jj*n_tot]<<endl;
      }
   }
   //
   output_test.close();

   return 0;
};
