#include "CEED_phon_subs.h"

//##############################################################################
void read_inputs(UNINT& n_el, UNINT& n_phon, UNINT& np_levels, UNINT& n_tot,
                 UNINT& n_bath, vector<double>& el_ener_vec,
                 vector<double>& w_phon_vec, vector<double>& mass_phon_vec,
                 vector<double>& fb_vec){

   ifstream inputf;

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
   inputf>> n_bath;

   for (int ii=0; ii<n_el; ii++){
      double el_ener;
      inputf>> el_ener;
      el_ener_vec.push_back(el_ener);
   }

   for (int ii=0; ii<n_el; ii++){
      double fb;
      inputf>> fb;
      fb_vec.push_back(fb);
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

   return;
}
//##############################################################################
void read_matrix_inputs(UNINT& n_el, UNINT& n_phon, UNINT& np_levels,
                        UNINT& n_tot, vector<double>& Fcoup_mat,
                        vector<double>& mu_elec_mat, vector<double>& dVdX_mat){

   int      n_inter;
   int      ind_i;
   int      ind_j;
   double   Mij;
   ifstream inputf;

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
      Fcoup_mat[ind_i + ind_j*n_el] = Mij;
      Fcoup_mat[ind_j + ind_i*n_el] = Mij;
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
                 UNINT n_bath){

   for(int ii=0; ii<(n_tot*n_tot); ii++){
      H0_mat.push_back(0.0e0);
      Hcoup_mat.push_back(0.0e0);
      mu_elec_mat.push_back(0.0e0);
      mu_phon_mat.push_back(0.0e0);
      v_bath_mat.push_back(0.0e0);
      eigen_coef.push_back(0.0e0);
      eigen_coefT.push_back(0.0e0);
      mu_tot.push_back((0.0e0, 0.0e0));
      H_tot.push_back((0.0e0, 0.0e0));
      rho_tot.push_back((0.0e0, 0.0e0));
      rho_new.push_back((0.0e0, 0.0e0));
   }

   for(int ii=0; ii<n_tot; ii++){
      eigen_E.push_back(0.0e0);
   }

   for(int ii=0; ii<(n_el*n_el); ii++){
      rho_elec.push_back((0.0e0, 0.0e0));
      Fcoup_mat.push_back(0.0e0);
   }

   for(int ii=0; ii<(n_phon*n_phon*np_levels*np_levels); ii++){
      dVdX_mat.push_back(0.0e0);
      rho_phon.push_back((0.0e0, 0.0e0));
   }

   for(int ii=0; ii<n_bath; ii++){
      xi_vec.push_back(0.0e0);
      ki_vec.push_back(0.0e0);
   }

   return;
}
//##############################################################################
void build_rho_matrix(vector < complex<double> >& rho_tot,
                      vector<double>& eigen_coef, vector<double>& eigen_coefT,
                      UNINT n_tot){

   vector<double> rho_real(n_tot*n_tot, 0.0e0);
   vector<double> auxmat1(n_tot*n_tot, 0.0e0);

   rho_real[1+1*n_tot] = 2.0e0;

   matmul_blas(rho_real, eigen_coefT, auxmat1, n_tot);
   matmul_blas(eigen_coef, auxmat1, rho_real, n_tot);

   for (int ii=0; ii < n_tot*n_tot; ii++){
      rho_tot[ii] = complex<double> (rho_real[ii],0.0e0);
   }

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
void build_matrix(vector < complex<double> >& H_tot, vector<double>& H0_mat,
                  vector<double>& Hcoup_mat, vector<double>& mu_phon_mat,
                  vector<double>& dVdX_mat, vector<double>& Fcoup_mat,
                  vector<double>& mu_elec_mat,
                  vector < complex<double> >& mu_tot,
                  vector<double>& el_ener_vec, vector<double>& w_phon_vec,
                  vector<double>& mass_phon_vec, double k0_inter, UNINT n_el,
                  UNINT n_phon, UNINT np_levels, UNINT n_tot){

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

   int ntemp = np_levels*n_phon;
   for (int ii=0; ii<ntemp; ii++){
   for (int jj=0; jj<ntemp; jj++){
      int ind1  = ii + jj*n_tot;
      int ind2  = ii + jj*ntemp;
      dVdX_mat[ind2] = k0_inter*mu_phon_mat[ind1];
      for (int k1=0; k1<n_el; k1++){
      for (int k2=0; k2<n_el; k2++){
         int ind3 = (ii + k1*ntemp) + (jj + k2*ntemp)*n_tot;
         Hcoup_mat[ind3] = -Fcoup_mat[k1 + k2*n_el] * mu_phon_mat[ind1];
      }
      }
   }
   }

   for (int ii=0; ii<n_tot*n_tot; ii++){
      H0_mat[ii]   += Hcoup_mat[ii];
      H_tot[ii]     = complex<double> (H0_mat[ii], 0.0e0);
      mu_tot[ii]    = complex<double> (mu_elec_mat[ii]+mu_phon_mat[ii], 0.0e0);
   }

   return;
}
//##############################################################################
void eigenval_elec_calc(vector<double>& mat, vector<double>& eigenval,
                        vector<double>& coef, UNINT ntotal){

   int          info, lwork;
   int          dim = (int) ntotal;
   UNINT n2  = ntotal * ntotal;
   double       wkopt;
   double*      work;
   char         jobz='V';
   char         uplo='U';

   for(int ii=0; ii < n2; ii++){coef[ii]=mat[ii];}

   lwork = -1;
   dsyev_( &jobz,&uplo, &dim, & *coef.begin(), &dim, & *eigenval.begin(),
           &wkopt, &lwork, &info);
   lwork = (int)wkopt;
   work = (double*)malloc( lwork*sizeof(double) );
   dsyev_( &jobz,&uplo, &dim, & *coef.begin(), &dim, & *eigenval.begin(), work,
           &lwork, &info );

   return;
}
//##############################################################################
void matmul_blas(vector<double>& matA, vector<double>& matB,
                 vector<double>& matC, int ntotal){

   char   trans = 'N';
   double alpha = 1.0e0;
   double beta  = 0.0e0;

   dgemm_(&trans, &trans, &ntotal, &ntotal, &ntotal, &alpha, & *matA.begin(),
          &ntotal, & *matB.begin(), &ntotal, &beta, & *matC.begin(), &ntotal);

   return;
}
//##############################################################################