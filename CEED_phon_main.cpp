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
   double                     CL_phot = 0.0e0;
   double                     CL_bath = 0.0e0;
   double                     w_max;
   double                     w_min;
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
   vector < complex<double> > chi_b1;
   vector < complex<double> > chi_b2;
   vector < complex<double> > chi_phot1;
   vector < complex<double> > chi_phot2;

   ofstream outfile[8], output_test;

   readinput(n_el, n_phon, np_levels, n_tot, t_steps, print_t, dt, Efield,
             b_temp, a_ceed, seed, k0_ke_inter, w_max, w_min, el_ener_vec,
             w_phon_vec, mass_phon_vec);
//Dirty thing:
   mass_bath = mass_phon_vec[0];

   init_matrix(H_tot, H0_mat, Hcoup_mat, Fcoup_mat, mu_elec_mat, mu_phon_mat,
               mu_tot, Efield_t, eigen_E, eigen_coef, eigen_coefT, rho_phon,
               rho_tot, n_tot, n_el, n_phon, np_levels,
               Fbath_mat, X_phon_mat, chi_b1, chi_b2, chi_phot1, chi_phot2);

   read_matrix_inputs(n_el, n_phon, np_levels, n_tot, Fcoup_mat, mu_elec_mat,
                      Fbath_mat, w_phon_vec, mass_phon_vec);
   readefield(efield_flag, efield_vec, w_noise_vec, A_noise_vec,
              phi_noise_vec, n_noise);
   build_matrix(H_tot, H0_mat, Hcoup_mat, mu_phon_mat, Fcoup_mat,
                mu_elec_mat, mu_tot, X_phon_mat, el_ener_vec,
                w_phon_vec, mass_phon_vec, n_el, n_phon, np_levels, n_tot);

   eigenval_elec_calc(H0_mat, eigen_E, eigen_coef, n_tot);

   for(int jj=0; jj < n_tot; jj++){
   for(int ii=0; ii < n_tot; ii++){
      eigen_coefT[jj + ii * n_tot] = eigen_coef[ii + jj * n_tot];
   }
   }


   init_chi_terms(CL_bath, CL_phot, X_phon_mat, mu_tot, chi_b1, chi_b2,
                  chi_phot1, chi_phot2, eigen_coef, eigen_coefT, eigen_E,
                  w_max, w_min, b_temp, mass_bath, k0_ke_inter, n_tot);

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
   // output_test.close();
//charly: we recover the diagonal element of H in eigen_E

   // for(int ii=0; ii < n_tot; ii++){
   //    eigen_E[ii] = H0_mat[ii+ii*n_tot];
   // }

   build_rho_matrix(rho_tot, eigen_coef, eigen_coefT, mass_phon_vec, w_phon_vec,
                    n_tot, n_el, n_phon, np_levels);


   init_cuda(& *H_tot.begin(), & *mu_tot.begin(), & *rho_tot.begin(),
             & *rho_phon.begin(), & *X_phon_mat.begin(), & *chi_b1.begin(),
             & *chi_b2.begin(), & *chi_phot1.begin(), & *chi_phot2.begin(),
             n_el, n_phon, np_levels, n_tot);

   init_output(outfile);
   write_output(mass_bath, dt, 0, print_t, n_el, n_phon, np_levels, n_tot,
                rho_tot, outfile);

   double k_ceed_aux = a_ceed;
   double CLb_aux = 0.0e0;
   double CLp_aux = 0.0e0;
//Here the time propagation beguin:---------------------------------------------
   for(int tt=1; tt<= t_steps; tt++){

      // if (dt*tt<1000 || dt*tt>20000){
         // CLb_aux = 0.0e0;
      // }
      // else{
         // CLb_aux = CL_bath;
      // }
      //
      // if (dt*tt>20000){
         a_ceed = k_ceed_aux;
         // CLp_aux = 1.0e7*CL_phot;
      // }
      // else{
         // CLp_aux = 0.0e0;
         // a_ceed = 0.0e0;
      // }

      efield_t(efield_flag, tt, dt, Efield, efield_vec, Efield_t,
               w_noise_vec, A_noise_vec, phi_noise_vec, n_noise);
      runge_kutta_propagator_cuda(mass_bath, a_ceed, dt, Efield_t[0],
                                  Efield_t[1], CLb_aux, CLp_aux, tt, n_el,
                                  n_phon, np_levels, n_tot);

      write_output(mass_bath, dt, tt, print_t, n_el, n_phon, np_levels, n_tot,
                   rho_tot, outfile);

   }
//------------------------------------------------------------------------------

//TESTING CUDA
   // commute_cuda(dev_Htot1, dev_rhotot, dev_rhonew, n_tot);

   // calcrhophon(dev_rhotot, n_el, n_phon, np_levels, n_tot);
   // getingmat(& *mu_tot.begin(), dev_mutot, n_tot);

   // double aux1_real = get_trace_cuda(dev_rhophon, np_levels*n_phon);
   // cout << aux1_real<<endl;

   // output_test.open("file.out");
   // for(int jj=0; jj<n_tot; jj++){
   //    for(int ii=0; ii<n_tot; ii++){
   //       output_test<<ii<<"  "<<jj<<"  "<<mu_tot[ii+jj*n_tot].real()<<endl;
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
