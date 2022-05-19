#include "cuda_subs.h"
#include "CEED_phon_subs.h"

int main(){

   UNINT                      n_el;
   UNINT                      n_phon;
   UNINT                      np_levels;
   UNINT                      n_tot;
   UNINT                      n_noise=0;
   int                        t_steps;
   int                        print_t;
   int                        seed;
   int                        efield_flag = -1;
   double                     dt;
   double                     k0_ke_inter;
   double                     sigma_ke;
   double                     a_ceed   = 2.0/3.0 * 1.0/pow(137.0,3.0);
   double                     Efield;
   double                     b_temp;
   double                     mass_bath;
   double                     C_term = 0.0e0;
   double                     LM_term = 0.0e0;
   vector<double>             el_ener_vec;
   vector<double>             w_phon_vec;
   vector<double>             mass_phon_vec;
   vector<double>             Fcoup_mat;
   vector<double>             Fbath_mat;
   vector<double>             mu_elec_mat;
   vector<double>             mu_phon_mat;
   vector<double>             H0_mat;
   vector<double>             Hcoup_mat;
   vector<double>             eigen_E;
   vector<double>             eigen_coef;
   vector<double>             eigen_coefT;
   vector<double>             efield_vec;
   vector<double>             Efield_t;
   vector<double>             mu_aux;
   vector<double>             A_noise_vec;
   vector<double>             w_noise_vec;
   vector<double>             phi_noise_vec;
   vector < complex<double> > H_tot;
   vector < complex<double> > mu_tot;
   vector < complex<double> > rho_phon;
   vector < complex<double> > rho_tot;
   vector < complex<double> > X_phon_mat;
   vector < complex<double> > P_phon_mat;

   ofstream outfile[8], output_test;

   readinput(n_el, n_phon, np_levels, n_tot, t_steps, print_t, dt, Efield,
             b_temp, a_ceed, seed, k0_ke_inter, el_ener_vec, w_phon_vec,
             mass_phon_vec);
//Dirty thing:
   mass_bath = mass_phon_vec[0];

   init_matrix(H_tot, H0_mat, Hcoup_mat, Fcoup_mat, mu_elec_mat, mu_phon_mat,
               mu_tot, Efield_t, eigen_E, eigen_coef, eigen_coefT, rho_phon,
               rho_tot, n_tot, n_el, n_phon, np_levels,
               Fbath_mat, X_phon_mat, P_phon_mat);

   read_matrix_inputs(n_el, n_phon, np_levels, n_tot, Fcoup_mat, mu_elec_mat,
                      Fbath_mat, w_phon_vec, mass_phon_vec);
   readefield(efield_flag, efield_vec, w_noise_vec, A_noise_vec,
              phi_noise_vec, n_noise);
   build_matrix(H_tot, H0_mat, Hcoup_mat, mu_phon_mat, Fcoup_mat,
                mu_elec_mat, mu_tot, X_phon_mat, P_phon_mat, el_ener_vec,
                w_phon_vec, mass_phon_vec, n_el, n_phon, np_levels, n_tot);

   eigenval_elec_calc(H0_mat, eigen_E, eigen_coef, n_tot);

   for(int jj=0; jj < n_tot; jj++){
   for(int ii=0; ii < n_tot; ii++){
      eigen_coefT[jj + ii * n_tot] = eigen_coef[ii + jj * n_tot];
   }
   }


   init_CL_terms(C_term, LM_term, b_temp, mass_bath, w_phon_vec[0],
                 k0_ke_inter);

   // vector<double> auxmat1(n_tot*n_tot, 0.0e0);
   // matmul_blas(mu_aux, eigen_coef, auxmat1, n_tot);
   // matmul_blas(eigen_coefT, auxmat1, mu_aux, n_tot);

   // output_test.open("mu_mat.out");
   // for(int ii=0; ii<n_tot; ii++){
   // for(int jj=0; jj<n_tot; jj++){
   //    output_test<<ii<<"  "<<jj<<"  "<<P_phon_mat[ii+jj*n_tot]<<endl;
   // }
   // }
   // for(int ii=0; ii<n_tot; ii++){
   //    output_test<<ii<<"  "<<pow(eigen_coef[ii+20*n_tot],2)<<endl;
   // }
   // for(int ii=0; ii<n_tot; ii++){
      // output_test<<eigen_E[ii]<<endl;
   // }
   // for(int ii=20; ii<n_tot; ii++){
   // for(int jj=20; jj<n_tot; jj++){
      // output_test<<ii<<"  "<<jj<<"  "<<H_tot[ii+jj*n_tot].real()<<endl;
   // }
   // }
   output_test.close();
//charly: we recover the diagonal element of H in eigen_E

   for(int ii=0; ii < n_tot; ii++){
      eigen_E[ii] = H0_mat[ii+ii*n_tot];
   }

   build_rho_matrix(rho_tot, eigen_coef, eigen_coefT, mass_phon_vec, w_phon_vec,
                    n_tot, n_el, n_phon, np_levels);


   init_cuda(& *H_tot.begin(), & *mu_tot.begin(), & *rho_tot.begin(),
             & *rho_phon.begin(), & *X_phon_mat.begin(), & *P_phon_mat.begin(),
             n_el, n_phon, np_levels, n_tot);

   init_output(outfile);
   write_output(mass_bath, dt, 0, print_t, n_el, n_phon, np_levels, n_tot,
                rho_tot, outfile);

   double k_ceed_aux = a_ceed;
   double L_aux = 0.0e0;
   double C_aux = 0.0e0;
//Here the time propagation beguin:---------------------------------------------
   for(int tt=1; tt<= t_steps; tt++){

      a_ceed = k_ceed_aux;

      // if (tt>40000){
      //    a_ceed = k_ceed_aux;
      //    L_aux  = 0.0e0;
      //    C_aux  = 0.0e0;
      // }
      // else{
      //    a_ceed = 0.0e0;
      //    L_aux  = LM_term;
      //    C_aux  = C_term;
      // }

      efield_t(efield_flag, tt, dt, Efield, efield_vec, Efield_t,
               w_noise_vec, A_noise_vec, phi_noise_vec, n_noise);
      runge_kutta_propagator_cuda(mass_bath, a_ceed, dt, Efield_t[0],
                                  Efield_t[1], C_aux, L_aux, tt, n_el,
                                  n_phon, np_levels, n_tot);

      write_output(mass_bath, dt, tt, print_t, n_el, n_phon, np_levels, n_tot,
                   rho_tot, outfile);

   }
//------------------------------------------------------------------------------

//TESTING CUDA
   // commute_cuda(dev_Htot1, dev_rhotot, dev_rhonew, n_tot);

   // calcrhophon(dev_rhotot, n_el, n_phon, np_levels, n_tot);
   // getingmat(& *H_tot.begin(), dev_Htot2, n_tot);

   // double aux1_real = get_trace_cuda(dev_rhophon, np_levels*n_phon);
   // cout << aux1_real<<endl;

   // output_test.open("file.out");
   // for(int jj=0; jj<n_tot; jj++){
   //    for(int ii=0; ii<n_tot; ii++){
   //       output_test<<ii<<"  "<<jj<<"  "<<mu_tot[ii+jj*n_tot]<<endl;
   //    }
   // }

   // output_test.open("testfile.out");
   //
   // for(int jj=0; jj<n_tot; jj++){
   //    for(int ii=0; ii<n_tot; ii++){
   //       output_test<<ii<<"  "<<jj<<"  "<<H_tot[ii+jj*n_tot].real() - H0_mat[ii+jj*n_tot]<<endl;
   //       }
   //    }
   // //
   // output_test.close();

   for (int ii=0; ii<8; ii++){
      outfile[ii].close();
   }
   free_cuda_memory();
   return 0;
};
