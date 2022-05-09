#include "CEED_phon_subs.h"
#include "cuda_subs.h"

//##############################################################################
void read_matrix_inputs(UNINT& n_el, UNINT& n_phon, UNINT& np_levels,
                        UNINT& n_tot, vector<double>& Fcoup_mat,
                        vector<double>& mu_elec_mat, vector<double>& Fbath_mat,
                        vector<double>& w_phon_vec,
                        vector<double>& mass_phon_vec){

   int      n_inter;
   int      ind_i;
   int      ind_j;
   double   Mij;
   ifstream inputf;
   vector<double> mu_elec_aux(n_el*n_el, 0.0e0);

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

   inputf.open("Fbath.in");
   if (!inputf) {
      cout << "Unable to open Fbath.in";
      exit(1); // terminate with error
   }

   inputf>> n_inter;
   for (int ii=0; ii<n_inter; ii++){
      inputf>> ind_i;
      inputf>> ind_j;
      inputf>> Mij;
      Fbath_mat[ind_i + ind_j*n_el] = Mij;
      Fbath_mat[ind_j + ind_i*n_el] = Mij;
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
      mu_elec_aux[ind_i + ind_j*n_el] = Mij;
      insert_eterm_in_bigmatrix(ind_i, ind_j, n_el, n_tot, n_phon, np_levels,
                                Mij, mu_elec_mat);
   }
   inputf.close();

// Inserting Franck Condon Terms:
   for (int e1=0; e1<n_el; e1++){
   for (int e2=e1; e2<n_el; e2++){
      double x0 = mu_elec_aux[e2+e2*n_el] - mu_elec_aux[e1+e1*n_el];
      if ((e1 != e2) && (x0 > 0.0e0)){
         for(int pp=0; pp<n_phon; pp++){
            for(int ii=0; ii<np_levels; ii++){
            for(int jj=0; jj<np_levels; jj++){
               int ind1 = ii + pp*np_levels + e1*np_levels*n_phon;
               int ind2 = jj + pp*np_levels + e2*np_levels*n_phon;
               int ind3 = ind1 + ind2*n_tot;
               int ind4 = ind2 + ind1*n_tot;
               double alpha = x0*sqrt(mass_phon_vec[pp]*w_phon_vec[pp]*0.5);
               mu_elec_mat[ind3] = mu_elec_aux[e1+e2*n_el]*
                                   franck_condon_term(alpha, ii, jj);
               mu_elec_mat[ind4] = mu_elec_mat[ind3];
            }
            }
         }
      }
   }
   }

   return;
}
//##############################################################################
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
                 vector<double>& Fbath_mat){

   for(int ii=0; ii<(n_tot*n_tot); ii++){
      H0_mat.push_back(0.0e0);
      Hcoup_mat.push_back(0.0e0);
      mu_elec_mat.push_back(0.0e0);
      mu_phon_mat.push_back(0.0e0);
      eigen_coef.push_back(0.0e0);
      eigen_coefT.push_back(0.0e0);
      mu_tot.push_back((0.0e0, 0.0e0));
      H_tot.push_back((0.0e0, 0.0e0));
      rho_tot.push_back((0.0e0, 0.0e0));
   }

   for(int ii=0; ii<n_tot; ii++){
      eigen_E.push_back(0.0e0);
      eta_s_vec.push_back(0.0e0);
      lambda_s_vec.push_back(0.0e0);
   }

   for(int ii=0; ii<(n_el*n_el); ii++){
      Fcoup_mat.push_back(0.0e0);
      Fbath_mat.push_back(0.0e0);
   }

   Efield_t.push_back(0.0e0);
   Efield_t.push_back(0.0e0);

   for(int ii=0; ii<(n_phon*n_phon*np_levels*np_levels); ii++){
      rho_phon.push_back((0.0e0, 0.0e0));
   }

   for (int ii=0; ii<n_ke_bath; ii++){
      w_ke_vec.push_back(0.0e0);
   }

   return;
}
//##############################################################################
void build_rho_matrix(vector < complex<double> >& rho_tot,
                      vector<double>& eigen_coef, vector<double>& eigen_coefT,
                      vector<double>& mass_phon_vec, vector<double>& w_phon_vec,
                      UNINT n_tot, UNINT n_el, UNINT n_phon, UNINT np_levels){

   vector<double> rho_real(n_tot*n_tot, 0.0e0);
   vector<double> auxmat1(n_tot*n_tot, 0.0e0);

//chrarly: creating intial coherent state
   double theta = 0.001;
   double dx    = -0.5*sqrt(mass_phon_vec[0]*w_phon_vec[0]*0.5);

   rho_real[0+0*n_tot] = pow(sin(theta),2);

   for (int e1=0; e1<n_el; e1++){
   for (int e2=e1; e2<n_el; e2++){
      // if ((e1 != e2) && (x0 > 0.0e0)){
         // for(int pp=0; pp<n_phon; pp++){
            for(int ii=0; ii<np_levels; ii++){
            for(int jj=0; jj<np_levels; jj++){
               int ind1 = ii + e1*np_levels*n_phon;
               int ind2 = jj + e2*np_levels*n_phon;
               int ind3 = ind1 + ind2*n_tot;
               int ind4 = ind2 + ind1*n_tot;
               double c1 = 0.0e0;
               double c2 = 0.0e0;

               if (e1==0 && ii==0){
                  c1 = sin(theta);
               }
               if (e2==0 && jj==0){
                  c2 = sin(theta);
               }

               if (e1==1){
                  c1 = cos(theta)*exp(-pow(dx,2)*0.5)*pow(dx,ii)/
                       sqrt(factorial(ii));
               }
               if (e2==1){
                  c2 = cos(theta)*exp(-pow(dx,2)*0.5)*pow(dx,jj)/
                       sqrt(factorial(jj));
               }

               rho_real[ind3] = c1*c2;
               rho_real[ind4] = c1*c2;
            }
            }
         // }
      // }
   }
   }

   // matmul_blas(rho_real, eigen_coefT, auxmat1, n_tot);
   // matmul_blas(eigen_coef, auxmat1, rho_real, n_tot);

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
      int ind1 = jj + ii*np_levels + ind_i*n_phon*np_levels;
      int ind2 = jj + ii*np_levels + ind_j*n_phon*np_levels;
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
                  vector<double>& Fcoup_mat,
                  vector<double>& mu_elec_mat,
                  vector < complex<double> >& mu_tot,
                  vector<double>& el_ener_vec, vector<double>& w_phon_vec,
                  vector<double>& mass_phon_vec, UNINT n_el,
                  UNINT n_phon, UNINT np_levels, UNINT n_tot){

   for (int ii=0; ii<n_el; ii++){
   for (int jj=0; jj<n_phon; jj++){
      double mwj = w_phon_vec[jj] * mass_phon_vec[jj];
      for (int kk=0; kk<np_levels; kk++){
         int ind1 = kk + jj*np_levels + ii*n_phon*np_levels;
         int ind2 = ind1 + (ind1 * n_tot);
         H0_mat[ind2]  = el_ener_vec[ii] + w_phon_vec[jj]*(0.5e0 + kk);
         // if (ii == 1){
         //    H0_mat[ind2]  += 0.5 * mass_phon_vec[jj] *
         //                     pow(w_phon_vec[jj]*0.5,2);
         // }
      }
      for (int kk=0; kk<np_levels-1; kk++){
         int ind1   = kk + jj*np_levels + ii*n_phon*np_levels;
         int ind2   = ind1 + ((ind1+1) * n_tot);
         int ind3   = ind1+1 + (ind1 * n_tot);
         double Mij = sqrt((kk+1.0)/(2.0*mwj));
         mu_phon_mat[ind2] += Mij;
         mu_phon_mat[ind3] += Mij;

         // if(ii == 1){
         //    H0_mat[ind2] = -mass_phon_vec[jj]*pow(w_phon_vec[jj],2)*0.5*Mij;
         //    H0_mat[ind3] = H0_mat[ind2];
         // }
      }
   }
   }

   int n1 = np_levels*n_phon;
   for (int ii=0; ii<n1; ii++){
   for (int jj=0; jj<n1; jj++){
      int ind1  = ii + jj*n_tot;
      int ind2  = ii + jj*n1;
      for (int k1=0; k1<n_el; k1++){
      for (int k2=0; k2<n_el; k2++){
         int ind3 = (ii + k1*n1) + (jj + k2*n1)*n_tot;
         Hcoup_mat[ind3] = -Fcoup_mat[k1 + k2*n_el] * mu_phon_mat[ind1];
      }
      }
   }
   }

   for (int ii=0; ii<n_tot*n_tot; ii++){
      H0_mat[ii]    += Hcoup_mat[ii];
      H_tot[ii]      = complex<double> (H0_mat[ii], 0.0e0);
      mu_tot[ii]     = complex<double> (mu_elec_mat[ii]+mu_phon_mat[ii], 0.0e0);
   }

   // for (int ii=0; ii<n_tot; ii++){
   // // for (int jj=0; jj<n_tot; jj++){
   //    cout<<ii<<"   "<<ii<<"   "<<H_tot[ii+ii*n_tot]<<endl;
   // // }
   // }

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
double drand(){
  return (rand()+1.0)/(RAND_MAX+1.0);
}
//##############################################################################
double rand_normal(){
  return sqrt(-2 * log(drand())) * cos(2*M_PI*drand());
}
//##############################################################################
double rand_gaussian(double mean, double stdev){
  return mean + stdev * rand_normal();
}
//##############################################################################
void init_bath(double temp, double bmass, double bfreq, double span, int seed,
               UNINT n_ke_bath, vector<double>& w_ke_vec){
   srand(seed);

   double stdev = sqrt(temp/bmass*8.6173324e-5/27.2113862);
   double ki;

   for(int ii=0; ii<n_ke_bath; ii++){
      ki         = (0.5 + drand()) * bfreq;
      w_ke_vec[ii] = ki;
   }
   return;
}
//##############################################################################
void init_output(ofstream* outfile){
    outfile[0].open("energy.out");
    outfile[1].open("dipole.out");
    outfile[2].open("time.out");
    outfile[3].open("rhotrace.out");
    outfile[4].open("rho_elec.out");
    outfile[5].open("rho_phon.out");
    outfile[6].open("total_rho.out");
    outfile[7].open("rho_mat.out");

    return;
}
//##############################################################################
void write_output(double mass_bath, double dt, int tt, int print_t, UNINT n_el,
                  UNINT n_phon, UNINT np_levels, UNINT n_tot,
                  vector < complex<double> >& rho_tot, ofstream* outfile){

   double Ener, mu, trace_rho;
   vector<double> tr_rho_el(n_el, 0.0e0);
   vector<double> tr_rho_ph(n_phon*np_levels, 0.0e0);
   vector< complex<double> > tr_rho(n_tot,0.0e0);

   if(tt%print_t==0){
      getting_printing_info( & Ener, & mu, & *tr_rho.begin(),
                             & *rho_tot.begin(), n_tot);

      outfile[0]<<Ener<<endl;
      outfile[1]<<mu<<endl;
      outfile[2]<< tt*dt <<endl;

      trace_rho = 0.0e0;
      for(int ii=0; ii<n_el; ii++){
      for(int jj=0; jj<n_phon*np_levels; jj++){
          int ind1 = jj + ii*n_phon*np_levels;
          tr_rho_el[ii] += tr_rho[ind1].real();
          tr_rho_ph[jj] += tr_rho[ind1].real();
          trace_rho     += tr_rho[ind1].real();
      }
      }

      outfile[3]<<trace_rho<<endl;

      for(int ii=0; ii<n_el; ii++){
         outfile[4]<<"   "<<ii<<"   "<<tr_rho_el[ii]<<endl;
      }
      for(int jj=0; jj<n_phon*np_levels; jj++){
         outfile[5]<<"   "<<jj<<"   "<<tr_rho_ph[jj]<<endl;
      }

//temporal para caso simple de 1 unioc fonon
      for(int kk=0; kk<n_el; kk++){
      for(int jj=0; jj<np_levels; jj++){
      for(int ii=jj; ii<np_levels; ii++){
         int ind1 = ii + kk * np_levels;
         int ind2 = jj + kk * np_levels;
         int ind3 = ind1 + ind2 * n_tot;
         outfile[7]<<rho_tot[ind3].real()<<endl;
      }
      }
      }

   }
   return;
}
//##############################################################################
void readinput(UNINT& n_el, UNINT& n_phon, UNINT& np_levels, UNINT& n_tot,
               int& t_steps, int& print_t, double& dt, double& Efield,
               double& b_temp, double& a_ceed, int& seed, UNINT& n_ke_bath,
               double& sigma_ke, double& k0_ke_inter,
               vector<double>& el_ener_vec, vector<double>& w_phon_vec,
               vector<double>& mass_phon_vec){

  ifstream inputf;
  inputf.open("input.in");
  if (!inputf) {
     cout << "Unable to open input.in";
     exit(1); // terminate with error
  }

  int eqpos;
  string str;
  double k_ceed;

  string keys[] = {"N_electrons",
    "N_phonons",
    "N_phon_levels",
    "K_ceed",
    "Field_amp",
    "Total_time",
    "Delta_t",
    "print_step",
    "bath_temp",
    "N_ke_bath",
    "sigma_ke",
    "k0_ke_inter",
    "seed"};

  string veckeys[] = {"Elec_levels",
    "Phon_freq",
    "Phon_mass"};

  while (getline(inputf, str))
  {
    //cout << str << "\n";
    for(int jj=0; jj<15; jj++)
    {
      size_t found = str.find(keys[jj]);
      if (found != string::npos)
      {
        stringstream   linestream(str);
        string         data;
        getline(linestream, data, '=');
        if(jj==0) linestream >> n_el;
        else if(jj==1) linestream >> n_phon;
        else if(jj==2) linestream >> np_levels;
        else if(jj==3) linestream >> k_ceed;
        else if(jj==4) linestream >> Efield;
        else if(jj==5) linestream >> t_steps;
        else if(jj==6) linestream >> dt;
        else if(jj==7) linestream >> print_t;
        else if(jj==8) linestream >> b_temp;
        else if(jj==9) linestream >> n_ke_bath;
        else if(jj==10) linestream >> sigma_ke;
        else if(jj==11) linestream >> k0_ke_inter;
        else if(jj==12) linestream >> seed;
      }
    }

  }

  n_tot = n_el * n_phon * np_levels;
  a_ceed = a_ceed * k_ceed;

  inputf.close();

  inputf.open("input.in");
  while (getline(inputf, str))
  {
    //cout << str << "\n";
    for(int jj=0; jj<3; jj++)
    {
      size_t found = str.find(veckeys[jj]);
      if (found != string::npos)
      {
        stringstream linestream(str);
        string         data;
        getline(linestream, data, '=');
        if(jj==0){
           for (int ii=0; ii<n_el; ii++){
              double el_ener;
              linestream >> el_ener;
              el_ener_vec.push_back(el_ener);
           }
        }
        else if(jj==1){
           for (int ii=0; ii<n_phon; ii++){
              double w_phon;
              linestream >> w_phon;
              w_phon_vec.push_back(w_phon);
           }
        }
        else if(jj==2){
           for (int ii=0; ii<n_phon; ii++){
              double mass_phon;
              linestream >> mass_phon;
              mass_phon_vec.push_back(mass_phon);
           }
        }
      }
    }

  }
  inputf.close();
  return;
}
//##############################################################################
void readefield(int& efield_flag, vector<double>& efield_vec){
   ifstream inputf;
   double val;
   inputf.open("efield.in");
   if (!inputf) {
      cout << "Unable to open efield.in";
      exit(1); // terminate with error
   }

   inputf >> efield_flag;
   for(int ii=0; ii<5; ii++){
      inputf >> val;
      efield_vec.push_back(val);
   }

   //check shapes
   if((efield_flag == 0) || (efield_flag == 2)){
      if((efield_vec[0]>efield_vec[1]) || (efield_vec[2]>efield_vec[3])){
         cout << "Bad start-end definition in efield.in";
         exit(1); // terminate with error
      }
   }
   if((efield_flag == 1) || (efield_flag == 3)){
      if(efield_vec[0]>efield_vec[1]){
         cout << "Bad start-end definition in efield.in";
         exit(1); // terminate with error
      }
      if((efield_vec[3]<=0) || (efield_vec[4]<=0)){
         cout << "Wrong definitions in efield.in";
         exit(1); // terminate with error
      }
   }
   return;
}
//##############################################################################

void efield_t(int efield_flag, int tt, double dt, double Efield,
                 vector<double> efield_vec, vector<double>& Efield_t){
   double time = dt * tt;
   double timeaux = time + dt/2.0;
   double phase;

   Efield_t[0] = Efield;
   Efield_t[1] = Efield;

   if(efield_flag == -1){
      cout << "Invalid electric field";
      exit(1);
   }
   else if(efield_flag == 0){
      if((time>efield_vec[0]) && (timeaux<efield_vec[1])){
         phase = (time - efield_vec[0])/(efield_vec[1]-efield_vec[0]) * M_PI;
         Efield_t[0] *= 0.5 * (1-cos(phase));
         phase = (timeaux - efield_vec[0])/(efield_vec[1]-efield_vec[0]) * M_PI;
         Efield_t[1] *= 0.5 * (1-cos(phase));
      }
      else if ((time>efield_vec[2]) && (timeaux<efield_vec[3])){
         phase = (time - efield_vec[2])/(efield_vec[3]-efield_vec[2]) * M_PI;
         Efield_t[0] *= 0.5 * (1+cos(phase));
         phase = (timeaux - efield_vec[2])/(efield_vec[3]-efield_vec[2]) * M_PI;
         Efield_t[1] *= 0.5 * (1+cos(phase));
      }
      else if((time<=efield_vec[0]) || (timeaux>=efield_vec[3])){
         Efield_t[0] = 0.0e0;
         Efield_t[1] = 0.0e0;
      }
   }
   else if(efield_flag == 1){
      if((timeaux>efield_vec[0]) && (timeaux<efield_vec[1])){
         phase = pow((time - efield_vec[2])/efield_vec[3], 2);
         Efield_t[0] *= exp(-0.5 * phase);
         phase = pow((timeaux - efield_vec[2])/efield_vec[3], 2);
         Efield_t[1] *= exp(-0.5 * phase);
      }
      else{
         Efield_t[0] = 0.0e0;
         Efield_t[1] = 0.0e0;
      }
   }
   else if(efield_flag == 2){
      Efield_t[0] *= sin(efield_vec[4] * time);
      Efield_t[1] *= sin(efield_vec[4] * timeaux);
      if((time>efield_vec[0]) && (timeaux<efield_vec[1])){
         phase = (time - efield_vec[0])/(efield_vec[1]-efield_vec[0]) * M_PI;
         Efield_t[0] *= 0.5 * (1-cos(phase));
         phase = (timeaux - efield_vec[0])/(efield_vec[1]-efield_vec[0]) * M_PI;
         Efield_t[1] *= 0.5 * (1-cos(phase));
      }
      else if ((time>efield_vec[2]) && (timeaux<efield_vec[3])){
         phase = (time - efield_vec[2])/(efield_vec[3]-efield_vec[2]) * M_PI;
         Efield_t[0] *= 0.5 * (1+cos(phase));
         phase = (timeaux - efield_vec[2])/(efield_vec[3]-efield_vec[2]) * M_PI;
         Efield_t[1] *= 0.5 * (1+cos(phase));
      }
      else if((time<=efield_vec[0]) || (timeaux>=efield_vec[3])){
         Efield_t[0] = 0.0e0;
         Efield_t[1] = 0.0e0;
      }
   }
   else if(efield_flag == 3){
      if((timeaux>efield_vec[0]) && (timeaux<efield_vec[1])){
         Efield_t[0] *= cos(efield_vec[4] * (time-efield_vec[2]));
         Efield_t[1] *= cos(efield_vec[4] * (timeaux-efield_vec[2]));
         phase = pow((time - efield_vec[2])/efield_vec[3], 2);
         Efield_t[0] *= exp(-0.5 * phase);
         phase = pow((timeaux - efield_vec[2])/efield_vec[3], 2);
         Efield_t[1] *= exp(-0.5 * phase);
      }
      else{
         Efield_t[0] = 0.0e0;
         Efield_t[1] = 0.0e0;
      }
   }
   return;
}
//##############################################################################
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
                      vector<double>& Fbath_mat){

   int nat2 = n_tot*n_tot;
   const double pi = 3.141592653589793;
   n_ke_inter = 0;

   for (int e1=0; e1<n_el; e1++){
   for (int e2=0; e2<n_el; e2++){
   for (int pp=0; pp<n_phon; pp++){
      for (int ii=0; ii<np_levels; ii++){
      for (int jj=0; jj<np_levels; jj++){
         int ind_i = ii + pp * np_levels + e1 * np_levels * n_phon;
         int ind_j = jj + pp * np_levels + e2 * np_levels * n_phon;
         double C_e  = 0.0e0;
         double C_ph = 0.0e0;
         if(ii == jj){
            C_e = Fbath_mat[e1, e2*n_el];
         }
         if((ii == jj+1) && e1==e2){
            C_ph = sqrt(0.5*1/(mass_phon_vec[pp]*w_phon_vec[pp]))*sqrt(jj+1);
         }
         else if(ii == jj-1 && e1==e2){
            C_ph = sqrt(0.5*1/(mass_phon_vec[pp]*w_phon_vec[pp]))*sqrt(jj);
         }
         for (int kk=0; kk<n_ke_bath; kk++){
            double aux1, exp1, exp2;
            double Ea   = eigen_E[ind_i];
            double Eb   = eigen_E[ind_j];
            double wj   = w_ke_vec[kk];
            aux1 = pow(C_e + k0_ke_inter*C_ph, 2.0) * pi / (wj * mass_bath);
            exp1 = dirac_delta(Ea, Eb, wj, sigma_ke);
            exp2 = dirac_delta(Ea, Eb, -wj, sigma_ke);
            bool acceptable;
            acceptable = (aux1*exp1 > 1.0e-6) || (aux1*exp2 > 1.0e-6);
            if (acceptable){
               ke_index_i.push_back(ind_i);
               ke_index_j.push_back(ind_j);
               ke_index_k.push_back(kk);
               ke_delta1_vec.push_back(aux1*exp1);
               ke_delta2_vec.push_back(aux1*exp2);
               eta_l_vec.push_back(0.0e0);
               lambda_l_vec.push_back(0.0e0);
               n_ke_inter += 1;
            }
         }

      }
      }
   }
   }
   }

   cout<<n_ke_inter<<endl;

   return;
}
//##############################################################################
void creating_ke_bath(UNINT n_ke_bath, vector<double>& w_ke_vec,
                      vector<double>& ke_N_phon_vec, double b_temp){

   double temp_Ha = b_temp*8.6173324e-5/27.2113862;
   for (int ii=0; ii<n_ke_bath; ii++){
      ke_N_phon_vec.push_back(1.0e0 / (exp(w_ke_vec[ii]/temp_Ha)-1.0e0));
   }

   return;
}

//##############################################################################
double dirac_delta(double Ea, double Eb, double wj, double sigma){

   double norm;
   double arg;
   double dir_del;
   const double pi = 3.141592653589793;

   norm = 1.0 / sqrt(2.0 * pi * pow(sigma, 2.0));
   arg  = -pow((Ea - Eb + wj), 2.0)/(2.0 * pow(sigma, 2.0));

   dir_del = norm * exp(arg);

   return dir_del;
}
//##############################################################################
double factorial(int n){
    if (n > 1) {
        return n * factorial(n - 1);
    } else {
        return 1;
    }
}
//##############################################################################
double franck_condon_term(double alpha, int ind_i, int ind_j){

   double FC_ij = 0.0e0;
   double num1 = sqrt(factorial(ind_j)*factorial(ind_i));
   double c1   = exp(-0.5*pow(alpha,2.0));

   if (ind_i >= ind_j){
      for (int ss=0; ss<=ind_j;ss++){
         double num2 = factorial(ss)*factorial(ss+ind_i-ind_j)*
                       factorial(ind_j-ss);
         int exp1 =  2*ss+(ind_i-ind_j);
         int exp2 = ss;

         FC_ij += c1 * pow(alpha,exp1) * pow(-1,exp2) * num1/num2;
      }
   }
   else if(ind_j > ind_i){
      for (int ss=0; ss<=ind_i;ss++){
         double num2 = factorial(ss)*factorial(ss+ind_j-ind_i)*
                       factorial(ind_i-ss);
         int exp1 =  2*ss+(ind_j-ind_i);
         int exp2 =  ss+(ind_j-ind_i);

         FC_ij += c1 * pow(alpha,exp1) * pow(-1,exp2) * num1/num2;
      }
   }

   return FC_ij;
}
//##############################################################################
