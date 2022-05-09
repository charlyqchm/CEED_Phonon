#include "cuda_subs.h"

cuDoubleComplex  *dev_rhophon;
cuDoubleComplex  *dev_rhotot;
cuDoubleComplex  *dev_rhonew;
cuDoubleComplex  *dev_rhoaux;
cuDoubleComplex  *dev_Drho;
cuDoubleComplex  *dev_Htot1;
cuDoubleComplex  *dev_Htot2;
cuDoubleComplex  *dev_Htot3;
cuDoubleComplex  *dev_mutot;
double           *dev_etaL_ke;
double           *dev_lambdaL_ke;
double           *dev_etaS_ke;
double           *dev_lambdaS_ke;
double           *dev_ke_del1;
double           *dev_ke_del2;
double           *dev_Nphon_ke;
int              *dev_keind_j;
int              *dev_keind_k;
UNINT             Ncores1;
UNINT             Ncores2;
UNINT             Ncores3;

//##############################################################################
// This function build the Hamiltonian without CEED:
// H_in  = H_e + H_phon + H_e-phon - This matrix is the same always.
// H_out = H_in + E * mu + V_bath
__global__ void update_H_tot(cuDoubleComplex *H_out, cuDoubleComplex *H_in,
                             cuDoubleComplex *mu_tot, double Efield, int n_el,
                             int n_phon, int np_levels, int n_tot){

   int ind  = threadIdx.x + blockIdx.x * blockDim.x;
   int dim2 = n_tot * n_tot;
   cuDoubleComplex aux1;

   if (ind < dim2){
      H_out[ind] = H_in[ind];

      aux1 = make_cuDoubleComplex(Efield, 0.0e0);
      aux1 = cuCmul(aux1, mu_tot[ind]);

      H_out[ind] = cuCadd(H_out[ind], aux1);
   }
   return;
}
//##############################################################################
//This function extract the diagonal terms of the matrix matA in vecA
__global__ void get_diag(cuDoubleComplex *matA, cuDoubleComplex *vecA,
                         int n_tot){

   int ind  = threadIdx.x + blockIdx.x * blockDim.x;
   int dim2 = n_tot  * n_tot;
   int i1   = ind / n_tot;
   if ((ind == i1 + i1*n_tot) && (ind < dim2)){
      vecA[i1] = matA[ind];
   }
   return;
}
//##############################################################################
__global__ void build_rhophon(cuDoubleComplex *rho_tot,
                              cuDoubleComplex *rho_phon, int n_el ,int n_phon,
                              int np_levels, int n_tot){

   int ind1 = threadIdx.x + blockIdx.x * blockDim.x;
   int dim1 = n_phon * np_levels;
   int dim2 = dim1 * dim1;

   if (ind1 < dim2){
      int jj = ind1/dim1;
      int ii = ind1 - jj * dim1;
      rho_phon[ind1] = make_cuDoubleComplex(0.0e0, 0.0e0);
      for (int kk=0; kk<n_el; kk++){
         int ind2 = (ii + kk * dim1) + (jj + kk * dim1) * n_tot;
         rho_phon[ind1] = cuCadd(rho_tot[ind2], rho_phon[ind1]);
      }
   }
   return;
}
//##############################################################################
__global__ void update_mat(cuDoubleComplex *matA, cuDoubleComplex *matB,
                           int dim){
   int ind  = threadIdx.x + blockIdx.x * blockDim.x;
   int dim2 = dim * dim;
   if (ind < dim2){
      matA[ind] = matB[ind];
   }
   return;
}
//##############################################################################
__global__ void get_long_ke_vectors(cuDoubleComplex *rho, double *ke_N_phon,
                                    double *eta_long, double *lambda_long,
                                    double *ke_del1, double *ke_del2,
                                    int *ke_ind_j, int *ke_ind_k,
                                    int n_tot, int n_ke_inter){

   int ind     = threadIdx.x + blockIdx.x * blockDim.x;

   if (ind < n_ke_inter){
      int jj      = ke_ind_j[ind];
      int kk      = ke_ind_k[ind];
      double del1 = ke_del1[ind];
      double del2 = ke_del2[ind];
      double f2   = cuCreal(rho[jj + jj * n_tot]);
      double Nj   = ke_N_phon[kk];

      eta_long[ind]    = -((Nj + f2) * del1 + (Nj - f2 + 1.0e0) * del2);
      lambda_long[ind] = f2*((Nj + 1.0e0) * del1 + Nj * del2);
   }

   return;
}
//##############################################################################
__global__ void apply_ke_term(cuDoubleComplex *rho, cuDoubleComplex *Drho,
                              double *eta_short, double *lambda_short,
                              int n_tot){

   int ind  = threadIdx.x + blockIdx.x * blockDim.x;
   int dim2 = n_tot  * n_tot;
   int i1   = ind / n_tot;
   int i2   = ind - i1*n_tot;
   if ((ind == i2 + i1*n_tot) && (ind < dim2)){
   // if ((ind == i1 + i1*n_tot) && (ind < dim2)){
      cuDoubleComplex aux1 = make_cuDoubleComplex(eta_short[i1], 0.0e0);
      cuDoubleComplex aux2 = make_cuDoubleComplex(eta_short[i2], 0.0e0);
      cuDoubleComplex aux3;
      cuDoubleComplex aux4 = make_cuDoubleComplex(0.5e0, 0.0e0);
      cuDoubleComplex aux5 = make_cuDoubleComplex(0.0e0,0.0e0);
      aux3 = cuCadd(aux1,aux2);
      aux3 = cuCmul(aux4,aux3);
      aux3 = cuCmul(aux3,rho[i2+i1*n_tot]);
      if (ind == i1 + i1*n_tot){
         aux5 = make_cuDoubleComplex(lambda_short[i1], 0.0e0);
      }
      aux5 = cuCadd(aux3,aux5);
      Drho[i2+i1*n_tot] = cuCadd(Drho[i2+i1*n_tot], aux5);
   }
   return;
}

//##############################################################################
void init_cuda(complex<double> *H_tot, complex<double> *mu_tot,
               complex<double> *rho_tot,
               complex<double> *rho_phon,
               int *ke_index_i, int *ke_index_j, int *ke_index_k,
               double *ke_delta1_vec, double *ke_delta2_vec,
               double *ke_N_phon_vec,
               UNINT n_el, UNINT n_phon, UNINT np_levels, UNINT n_tot,
               UNINT n_ke_bath, UNINT n_ke_inter){

   double gaux = (double) (n_tot*n_tot);
   double taux = (double) Nthreads;

   Ncores1 = (UNINT) ceil(gaux/taux);
   gaux    = (double) (n_ke_inter);
   Ncores3 = (UNINT) ceil(gaux/taux);

   int dimaux  = n_phon * n_phon * np_levels * np_levels;

   cudaMalloc((void**) &dev_Htot1  , n_tot*n_tot*sizeof(cuDoubleComplex));
   cudaMalloc((void**) &dev_Htot2  , n_tot*n_tot*sizeof(cuDoubleComplex));
   cudaMalloc((void**) &dev_Htot3  , n_tot*n_tot*sizeof(cuDoubleComplex));
   cudaMalloc((void**) &dev_mutot  , n_tot*n_tot*sizeof(cuDoubleComplex));
   cudaMalloc((void**) &dev_rhotot , n_tot*n_tot*sizeof(cuDoubleComplex));
   cudaMalloc((void**) &dev_rhonew , n_tot*n_tot*sizeof(cuDoubleComplex));
   cudaMalloc((void**) &dev_rhoaux , n_tot*n_tot*sizeof(cuDoubleComplex));
   cudaMalloc((void**) &dev_Drho   , n_tot*n_tot*sizeof(cuDoubleComplex));
   cudaMalloc((void**) &dev_rhophon, dimaux*sizeof(cuDoubleComplex));
   cudaMalloc((void**) &dev_etaL_ke   , n_ke_inter*sizeof(double));
   cudaMalloc((void**) &dev_lambdaL_ke, n_ke_inter*sizeof(double));
   cudaMalloc((void**) &dev_etaS_ke   , n_tot*sizeof(double));
   cudaMalloc((void**) &dev_lambdaS_ke, n_tot*sizeof(double));
   cudaMalloc((void**) &dev_ke_del1   , n_ke_inter*sizeof(double));
   cudaMalloc((void**) &dev_ke_del2   , n_ke_inter*sizeof(double));
   cudaMalloc((void**) &dev_Nphon_ke  , n_ke_bath*sizeof(double));
   cudaMalloc((void**) &dev_keind_j   , n_ke_inter*sizeof(int));
   cudaMalloc((void**) &dev_keind_k   , n_ke_inter*sizeof(int));


   cudaMemcpy(dev_Htot1, H_tot, n_tot*n_tot*sizeof(cuDoubleComplex),
              cudaMemcpyHostToDevice);
   cudaMemcpy(dev_mutot, mu_tot, n_tot*n_tot*sizeof(cuDoubleComplex),
              cudaMemcpyHostToDevice);
   cudaMemcpy(dev_rhotot, rho_tot, n_tot*n_tot*sizeof(cuDoubleComplex),
              cudaMemcpyHostToDevice);
   cudaMemcpy(dev_rhophon, rho_phon, dimaux*sizeof(cuDoubleComplex),
              cudaMemcpyHostToDevice);
   cudaMemcpy(dev_ke_del1, ke_delta1_vec, n_ke_inter*sizeof(double),
              cudaMemcpyHostToDevice);
   cudaMemcpy(dev_ke_del2, ke_delta2_vec, n_ke_inter*sizeof(double),
              cudaMemcpyHostToDevice);
   cudaMemcpy(dev_Nphon_ke, ke_N_phon_vec, n_ke_bath*sizeof(double),
              cudaMemcpyHostToDevice);
   cudaMemcpy(dev_keind_j, ke_index_j, n_ke_inter*sizeof(int),
              cudaMemcpyHostToDevice);
   cudaMemcpy(dev_keind_k, ke_index_k, n_ke_inter*sizeof(int),
              cudaMemcpyHostToDevice);

   return;
}
//##############################################################################
void free_cuda_memory(){

   cudaFree(dev_Htot1);
   cudaFree(dev_Htot2);
   cudaFree(dev_Htot3);
   cudaFree(dev_mutot);
   cudaFree(dev_rhotot);
   cudaFree(dev_rhonew);
   cudaFree(dev_rhoaux);
   cudaFree(dev_Drho);
   cudaFree(dev_rhophon);
   cudaFree(dev_etaL_ke);
   cudaFree(dev_lambdaL_ke);
   cudaFree(dev_etaS_ke);
   cudaFree(dev_lambdaS_ke);
   cudaFree(dev_ke_del1);
   cudaFree(dev_ke_del2);
   cudaFree(dev_Nphon_ke);
   cudaFree(dev_keind_j);
   cudaFree(dev_keind_k);

   return;
}
//##############################################################################
void matmul_cublas(cuDoubleComplex *dev_A, cuDoubleComplex *dev_B,
                   cuDoubleComplex *dev_C, int dim){

   const cuDoubleComplex alf = make_cuDoubleComplex(1.0,0.0);
   const cuDoubleComplex bet = make_cuDoubleComplex(0.0, 0.0);
   const cuDoubleComplex *alpha = &alf;
   const cuDoubleComplex *beta = &bet;
// Create a handle for CUBLAS
   cublasHandle_t handle;
   cublasCreate(&handle);
// Do the actual multiplication
    cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, alpha, dev_A,
                dim, dev_B, dim, beta, dev_C, dim);
// Destroy the handle
    cublasDestroy(handle);
    return;

}
//##############################################################################
void commute_cuda(cuDoubleComplex *dev_A, cuDoubleComplex *dev_B,
                  cuDoubleComplex *dev_C, int dim, const cuDoubleComplex alf){

   const cuDoubleComplex bet1 = make_cuDoubleComplex(0.0, 0.0);
   const cuDoubleComplex bet2 = make_cuDoubleComplex(-1.0, 0.0);
   const cuDoubleComplex *alpha = &alf;
   const cuDoubleComplex *beta1 = &bet1;
   const cuDoubleComplex *beta2 = &bet2;
// Create a handle for CUBLAS
   cublasHandle_t handle;
   cublasCreate(&handle);
// Computing B.A
   cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, alpha, dev_B,
               dim, dev_A, dim, beta1, dev_C, dim);
// Computing A.B - B.A
   cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, alpha, dev_A,
               dim, dev_B, dim, beta2, dev_C, dim);
// Destroy the handle
   cublasDestroy(handle);
   return;
}
//##############################################################################
void matadd_cublas(cuDoubleComplex *dev_A, cuDoubleComplex *dev_B,
                   cuDoubleComplex *dev_C, int dim, const cuDoubleComplex alf,
                   const cuDoubleComplex bet){

   const cuDoubleComplex *alpha = &alf;
   const cuDoubleComplex *beta = &bet;
// Create a handle for CUBLAS
   cublasHandle_t handle;
   cublasCreate(&handle);
// Do the actual multiplication
    cublasZgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, alpha, dev_A,
                dim, beta ,dev_B, dim, dev_C, dim);
// Destroy the handle
    cublasDestroy(handle);
    return;
}
//##############################################################################
double get_trace_cuda(cuDoubleComplex *dev_A, UNINT dim){

   // cuDoubleComplex aux1= make_cuDoubleComplex(0.0e0, 0.0e0);
   complex<double> aux1;
   double          aux2;
   complex<double> aux_vec[dim];
   cuDoubleComplex *dev_vec;

   cudaMalloc((void**) &dev_vec, dim * sizeof(cuDoubleComplex));

   get_diag<<<Ncores1, Nthreads>>>(dev_A, dev_vec, dim);

   cudaMemcpy(aux_vec, dev_vec, dim*sizeof(cuDoubleComplex),
              cudaMemcpyDeviceToHost);

   for(int ii=0;ii<dim;ii++){
      aux1 += aux_vec[ii];
   }

   aux2 = aux1.real();

   cudaFree(dev_vec);
   return aux2;
}
//##############################################################################
void include_Hceed_cuda(cuDoubleComplex *dev_Hout, cuDoubleComplex *dev_Hin,
                        cuDoubleComplex *dev_mu, cuDoubleComplex *dev_rhoin,
                        double a_ceed, int n_tot){

   int dim2 = n_tot*n_tot;
   double dmu2;
   cuDoubleComplex *dev_aux1, *dev_aux2, *dev_Hceed;
   const cuDoubleComplex alf  = make_cuDoubleComplex(1.0,0.0);

   cudaMalloc((void**) &dev_aux1, dim2 * sizeof(cuDoubleComplex));
   cudaMalloc((void**) &dev_aux2, dim2 * sizeof(cuDoubleComplex));
   cudaMalloc((void**) &dev_Hceed, dim2 * sizeof(cuDoubleComplex));

   commute_cuda(dev_mu, dev_Hin, dev_Hceed, n_tot, alf);
   commute_cuda(dev_Hceed, dev_Hin, dev_aux1, n_tot, alf);
   matmul_cublas(dev_rhoin, dev_aux1, dev_aux2, n_tot);

   dmu2 = get_trace_cuda(dev_aux2, n_tot);

   const cuDoubleComplex bet = make_cuDoubleComplex(0.0, a_ceed*dmu2);

   matadd_cublas(dev_Hin, dev_Hceed, dev_Hout, n_tot, alf, bet);

   cudaFree(dev_aux1);
   cudaFree(dev_aux2);
   cudaFree(dev_Hceed);

   return;
}
//##############################################################################
void include_ke_terms(cuDoubleComplex *dev_rho, cuDoubleComplex *dev_drdt,
                      int *ke_index_i, double *eta_s_vec,
                      double *lambda_s_vec, double *eta_l_vec,
                      double *lambda_l_vec, UNINT n_tot, UNINT n_ke_inter){

      get_long_ke_vectors<<<Ncores3, Nthreads>>>(dev_rho, dev_Nphon_ke,
                            dev_etaL_ke, dev_lambdaL_ke, dev_ke_del1,
                            dev_ke_del2, dev_keind_j, dev_keind_k,
                            n_tot, n_ke_inter);

      cudaMemcpy(eta_l_vec, dev_etaL_ke, n_ke_inter*sizeof(double),
                 cudaMemcpyDeviceToHost);
      cudaMemcpy(lambda_l_vec, dev_lambdaL_ke, n_ke_inter*sizeof(double),
                 cudaMemcpyDeviceToHost);

      for(int ii=0; ii<n_tot; ii++){
         eta_s_vec[ii] = 0.0e0;
         lambda_s_vec[ii] = 0.0e0;
      }

      for(int ii=0; ii<n_ke_inter; ii++){
         int ind_i = ke_index_i[ii];

         eta_s_vec[ind_i]    += eta_l_vec[ii];
         lambda_s_vec[ind_i] += lambda_l_vec[ii];
      }

      // printf("%f",lambda_s_vec[11]);

      cudaMemcpy(dev_etaS_ke, eta_s_vec, n_tot*sizeof(double),
                 cudaMemcpyHostToDevice);
      cudaMemcpy(dev_lambdaS_ke, lambda_s_vec, n_tot*sizeof(double),
                 cudaMemcpyHostToDevice);

      apply_ke_term<<<Ncores1, Nthreads>>>(dev_rho, dev_drdt, dev_etaS_ke,
                                           dev_lambdaS_ke, n_tot);

      return;
}
//##############################################################################
void runge_kutta_propagator_cuda(double mass_bath, double a_ceed, double dt,
                                 double Efield, double Efieldaux,
                                 double *eta_s_vec,
                                 double *lambda_s_vec, double *eta_l_vec,
                                 double *lambda_l_vec, int *ke_index_i,
                                 int tt, UNINT n_el,
                                 UNINT n_phon, UNINT np_levels,
                                 UNINT n_tot, UNINT n_ke_inter){

   const cuDoubleComplex alf1 = make_cuDoubleComplex(0.5*dt,0.0e0);
   const cuDoubleComplex alf2 = make_cuDoubleComplex(dt, 0.0e0);
   const cuDoubleComplex alf3 = make_cuDoubleComplex(1.0e0, 0.0e0);
   const cuDoubleComplex alf4 = make_cuDoubleComplex(0.0e0, -1.0e0);
   //double time = dt * tt;

   //Efield_t = Efield * exp(-pow(((time-10.0)/0.2),2.0));

   //Building the new Hamiltonian at time = t ----------------------------------
   update_H_tot<<<Ncores1, Nthreads>>>(dev_Htot2, dev_Htot1, dev_mutot,
                                       Efield, n_el, n_phon, np_levels, n_tot);
   //Including CEED Hamiltonian:
   include_Hceed_cuda(dev_Htot3, dev_Htot2, dev_mutot, dev_rhotot, a_ceed,
                      n_tot);
   //---------------------------------------------------------------------------

   //Calculating rho(t+dt/2) using LvN------------------------------------------
   commute_cuda(dev_Htot3, dev_rhotot, dev_Drho, n_tot, alf4);
   include_ke_terms(dev_rhotot, dev_Drho, ke_index_i, eta_s_vec, lambda_s_vec,
                    eta_l_vec, lambda_l_vec, n_tot, n_ke_inter);
   matadd_cublas(dev_rhotot, dev_Drho, dev_rhoaux, n_tot, alf3, alf1);
   //---------------------------------------------------------------------------
   //Hencefort we repeat everything to obtain everything in t + dt -------------

   //Efield_t = Efield * exp(-pow(((time+dth-10.0)/0.2),2.0));

   update_H_tot<<<Ncores1, Nthreads>>>(dev_Htot2, dev_Htot1, dev_mutot,
                                       Efieldaux, n_el, n_phon, np_levels,
                                       n_tot);

   include_Hceed_cuda(dev_Htot3, dev_Htot2, dev_mutot, dev_rhoaux, a_ceed,
                      n_tot);

   commute_cuda(dev_Htot3, dev_rhoaux, dev_Drho, n_tot, alf4);
   include_ke_terms(dev_rhoaux, dev_Drho, ke_index_i, eta_s_vec, lambda_s_vec,
                    eta_l_vec, lambda_l_vec, n_tot, n_ke_inter);
   matadd_cublas(dev_rhotot, dev_Drho, dev_rhonew, n_tot, alf3, alf2);

   //---------------------------------------------------------------------------
   //We update rho, x and v:
   update_mat<<<Ncores1, Nthreads>>>(dev_rhotot, dev_rhonew, n_tot);

   return;
}
//##############################################################################
void calcrhophon(cuDoubleComplex *dev_rhoin, int n_el, int n_phon,
                 int np_levels, int n_tot){
   build_rhophon<<<Ncores1, Nthreads>>>(dev_rhoin, dev_rhophon, n_el , n_phon,
                                        np_levels, n_tot);
   return;
}
//##############################################################################
void getingmat(complex<double> *matA, cuDoubleComplex *dev_A, int n_tot){

   cudaMemcpy(matA, dev_A, n_tot*n_tot*sizeof(cuDoubleComplex),
              cudaMemcpyDeviceToHost);
   return;
}
//##############################################################################
void getting_printing_info(double *Ener, double *mu, complex<double> *tr_rho,
                           complex<double> *rho_tot, UNINT n_tot){

   int dim2 = n_tot * n_tot;
   cuDoubleComplex *dev_aux1;
   cuDoubleComplex *dev_vec;

   cudaMalloc((void**) &dev_aux1, dim2 * sizeof(cuDoubleComplex));
   cudaMalloc((void**) &dev_vec, n_tot * sizeof(cuDoubleComplex));

   matmul_cublas(dev_rhotot, dev_Htot1, dev_aux1, n_tot);
   *Ener = get_trace_cuda(dev_aux1, n_tot);

   matmul_cublas(dev_rhotot, dev_mutot, dev_aux1, n_tot);
   *mu = get_trace_cuda(dev_aux1, n_tot);

   get_diag<<<Ncores1, Nthreads>>>(dev_rhotot, dev_vec, n_tot);

   cudaMemcpy(tr_rho, dev_vec, n_tot*sizeof(cuDoubleComplex),
   cudaMemcpyDeviceToHost);

   cudaMemcpy(rho_tot, dev_rhotot, n_tot*n_tot*sizeof(cuDoubleComplex),
   cudaMemcpyDeviceToHost);

   cudaFree(dev_vec);
   cudaFree(dev_aux1);

   return;
}
//##############################################################################
