#include "cuda_subs.h"
#include "CEED_phon_subs.h"

int main(){

   UNINT                      n_el;
   UNINT                      n_phon;
   UNINT                      np_levels;
   UNINT                      n_tot;
   UNINT                      n_ke_bath;
   UNINT                      n_ke_inter;
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
   vector<int>                ke_index_i;
   vector<int>                ke_index_j;
   vector<int>                ke_index_k;
   vector<double>             ke_delta1_vec;
   vector<double>             ke_delta2_vec;
   vector<double>             ke_N_phon_vec;
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
   vector<double>             w_ke_vec;
   vector<double>             eta_l_vec;
   vector<double>             lambda_l_vec;
   vector<double>             eta_s_vec;
   vector<double>             lambda_s_vec;
   vector<double>             mu_aux;
   vector < complex<double> > H_tot;
   vector < complex<double> > mu_tot;
   vector < complex<double> > rho_phon;
   vector < complex<double> > rho_tot;

   ofstream outfile[8], output_test;

   readinput(n_el, n_phon, np_levels, n_tot, t_steps, print_t, dt, Efield,
             b_temp, a_ceed, seed, n_ke_bath, sigma_ke, k0_ke_inter,
             el_ener_vec, w_phon_vec, mass_phon_vec);
//Dirty thing:
   mass_bath = mass_phon_vec[0];

   init_matrix(H_tot, H0_mat, Hcoup_mat, Fcoup_mat, mu_elec_mat, mu_phon_mat,
               mu_tot, Efield_t, eigen_E, eigen_coef, eigen_coefT, rho_phon,
               rho_tot, n_tot, n_el, n_phon, np_levels, n_ke_bath, w_ke_vec,
               eta_s_vec, lambda_s_vec, Fbath_mat);

   read_matrix_inputs(n_el, n_phon, np_levels, n_tot, Fcoup_mat, mu_elec_mat,
                      Fbath_mat, w_phon_vec, mass_phon_vec);
   readefield(efield_flag, efield_vec);
   build_matrix(H_tot, H0_mat, Hcoup_mat, mu_phon_mat, Fcoup_mat,
                mu_elec_mat, mu_tot, el_ener_vec, w_phon_vec,
                mass_phon_vec, n_el, n_phon, np_levels, n_tot);

   eigenval_elec_calc(H0_mat, eigen_E, eigen_coef, n_tot);

   for(int jj=0; jj < n_tot; jj++){
   for(int ii=0; ii < n_tot; ii++){
      eigen_coefT[jj + ii * n_tot] = eigen_coef[ii + jj * n_tot];
   }
   }

   // for (int ii=0; ii < n_tot*n_tot; ii++){
      // mu_aux.push_back(mu_tot[ii].real());
   // }

   // vector<double> auxmat1(n_tot*n_tot, 0.0e0);
   // matmul_blas(mu_aux, eigen_coef, auxmat1, n_tot);
   // matmul_blas(eigen_coefT, auxmat1, mu_aux, n_tot);

   // output_test.open("mu_mat.out");
   // for(int ii=0; ii<n_tot; ii++){
   // for(int jj=0; jj<n_tot; jj++){
      // output_test<<ii<<"  "<<jj<<"  "<<mu_aux[ii+jj*n_tot]<<endl;
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

   for(int ii=0; ii < n_tot; ii++){
      eigen_E[ii] = H0_mat[ii+ii*n_tot];
   }

   build_rho_matrix(rho_tot, eigen_coef, eigen_coefT, mass_phon_vec, w_phon_vec,
                    n_tot, n_el, n_phon, np_levels);

   init_bath(b_temp, mass_bath, w_phon_vec[0], 1, seed, n_ke_bath, w_ke_vec);

   creating_ke_bath(n_ke_bath, w_ke_vec, ke_N_phon_vec, b_temp);

   getting_ke_terms(n_tot, n_ke_bath, n_ke_inter, n_el, n_phon, np_levels,
                    mass_bath, ke_index_i, ke_index_j, ke_index_k, sigma_ke,
                    k0_ke_inter, eigen_E, w_ke_vec,
                    ke_delta1_vec, ke_delta2_vec, eta_l_vec, lambda_l_vec,
                    w_phon_vec, mass_phon_vec, Fbath_mat);

   init_cuda(& *H_tot.begin(), & *mu_tot.begin(), & *rho_tot.begin(),
             & *rho_phon.begin(), & *ke_index_i.begin(), & *ke_index_j.begin(),
             & *ke_index_k.begin(), & *ke_delta1_vec.begin(),
             & *ke_delta2_vec.begin(), & *ke_N_phon_vec.begin(),
             n_el, n_phon, np_levels, n_tot, n_ke_bath, n_ke_inter);

   init_output(outfile);
   write_output(mass_bath, dt, 0, print_t, n_el, n_phon, np_levels, n_tot,
                rho_tot, outfile);

   double k_ceed_aux = a_ceed;
//Here the time propagation beguin:---------------------------------------------
   for(int tt=1; tt<= t_steps; tt++){

      if (tt>300000){
         a_ceed = k_ceed_aux;
      }
      else{
         a_ceed = 0.0e0;
      }

      efield_t(efield_flag, tt, dt, Efield, efield_vec, Efield_t);
      runge_kutta_propagator_cuda(mass_bath, a_ceed, dt, Efield_t[0],
                                  Efield_t[1], & *eta_s_vec.begin(),
                                  & *lambda_s_vec.begin(), & *eta_l_vec.begin(),
                                  & *lambda_l_vec.begin(),
                                  & *ke_index_i.begin(), tt, n_el, n_phon,
                                  np_levels, n_tot, n_ke_inter);

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
