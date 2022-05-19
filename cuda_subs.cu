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
cuDoubleComplex  *dev_Xmat;
cuDoubleComplex  *dev_Pmat;
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
void init_cuda(complex<double> *H_tot, complex<double> *mu_tot,
               complex<double> *rho_tot,
               complex<double> *rho_phon,
               complex<double> *X_phon_mat,
               complex<double> *P_phon_mat,
               UNINT n_el, UNINT n_phon, UNINT np_levels, UNINT n_tot){

   double gaux = (double) (n_tot*n_tot);
   double taux = (double) Nthreads;

   Ncores1 = (UNINT) ceil(gaux/taux);

   int dimaux  = n_phon * n_phon * np_levels * np_levels;

   cudaMalloc((void**) &dev_Htot1  , n_tot*n_tot*sizeof(cuDoubleComplex));
   cudaMalloc((void**) &dev_Htot2  , n_tot*n_tot*sizeof(cuDoubleComplex));
   cudaMalloc((void**) &dev_Htot3  , n_tot*n_tot*sizeof(cuDoubleComplex));
   cudaMalloc((void**) &dev_mutot  , n_tot*n_tot*sizeof(cuDoubleComplex));
   cudaMalloc((void**) &dev_rhotot , n_tot*n_tot*sizeof(cuDoubleComplex));
   cudaMalloc((void**) &dev_rhonew , n_tot*n_tot*sizeof(cuDoubleComplex));
   cudaMalloc((void**) &dev_rhoaux , n_tot*n_tot*sizeof(cuDoubleComplex));
   cudaMalloc((void**) &dev_Drho   , n_tot*n_tot*sizeof(cuDoubleComplex));
   cudaMalloc((void**) &dev_Xmat   , n_tot*n_tot*sizeof(cuDoubleComplex));
   cudaMalloc((void**) &dev_Pmat   , n_tot*n_tot*sizeof(cuDoubleComplex));
   cudaMalloc((void**) &dev_rhophon, dimaux*sizeof(cuDoubleComplex));

   cudaMemcpy(dev_Htot1, H_tot, n_tot*n_tot*sizeof(cuDoubleComplex),
              cudaMemcpyHostToDevice);
   cudaMemcpy(dev_mutot, mu_tot, n_tot*n_tot*sizeof(cuDoubleComplex),
              cudaMemcpyHostToDevice);
   cudaMemcpy(dev_rhotot, rho_tot, n_tot*n_tot*sizeof(cuDoubleComplex),
              cudaMemcpyHostToDevice);
   cudaMemcpy(dev_rhophon, rho_phon, dimaux*sizeof(cuDoubleComplex),
              cudaMemcpyHostToDevice);
   cudaMemcpy(dev_Xmat, X_phon_mat, n_tot*n_tot*sizeof(cuDoubleComplex),
              cudaMemcpyHostToDevice);
   cudaMemcpy(dev_Pmat, P_phon_mat, n_tot*n_tot*sizeof(cuDoubleComplex),
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
   cudaFree(dev_Xmat);
   cudaFree(dev_Pmat);

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
void anticommute_cuda(cuDoubleComplex *dev_A, cuDoubleComplex *dev_B,
                      cuDoubleComplex *dev_C, int dim,
                      const cuDoubleComplex alf){

   const cuDoubleComplex bet1 = make_cuDoubleComplex(0.0, 0.0);
   const cuDoubleComplex bet2 = make_cuDoubleComplex(1.0, 0.0);
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
// Do the actual addition
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
void include_noise_dumping(cuDoubleComplex *dev_rho, cuDoubleComplex *dev_drdt,
                           double LM_term, double C_term, int n_tot){

   int dim2 = n_tot*n_tot;
   cuDoubleComplex *dev_aux1, *dev_auxC, *dev_auxL;
   const cuDoubleComplex alf1   = make_cuDoubleComplex(1.0,0.0);
   const cuDoubleComplex alf2   = make_cuDoubleComplex(0.5,0.0);
   const cuDoubleComplex alf_C  = make_cuDoubleComplex(-C_term, 0.00);
   const cuDoubleComplex alf_L  = make_cuDoubleComplex(0.00, -LM_term);

   cudaMalloc((void**) &dev_aux1, dim2 * sizeof(cuDoubleComplex));
   cudaMalloc((void**) &dev_auxC, dim2 * sizeof(cuDoubleComplex));
   cudaMalloc((void**) &dev_auxL, dim2 * sizeof(cuDoubleComplex));

//Calculating -C/\hbar^2 [X,[X,\rho]]
   commute_cuda(dev_Xmat, dev_rho, dev_aux1, n_tot, alf1);
   commute_cuda(dev_Xmat, dev_aux1, dev_auxC, n_tot, alf_C);

//Calculating -i L/ (\hbar M) [X, 0.5 {P, \rho}]
   anticommute_cuda(dev_Pmat, dev_rho, dev_aux1, n_tot, alf2);
   commute_cuda(dev_Xmat, dev_aux1, dev_auxL, n_tot, alf_L);

//Adding C and L term to d\rho/dt
   matadd_cublas(dev_drdt, dev_auxC, dev_aux1, n_tot, alf1, alf1);
   matadd_cublas(dev_aux1, dev_auxL, dev_drdt, n_tot, alf1, alf1);

   cudaFree(dev_aux1);
   cudaFree(dev_auxL);
   cudaFree(dev_auxC);

   return;
}

//##############################################################################
void runge_kutta_propagator_cuda(double mass_bath, double a_ceed, double dt,
                                 double Efield, double Efieldaux,
                                 double C_term, double LM_term,
                                 int tt, UNINT n_el,
                                 UNINT n_phon, UNINT np_levels,
                                 UNINT n_tot){

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
   commute_cuda(dev_Htot3, dev_rhotot, dev_Drho, n_tot, alf4); // -i[H,\rho]
   // include_noise_dumping(dev_rhotot, dev_Drho, LM_term, C_term, n_tot);

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
   // include_noise_dumping(dev_rhoaux, dev_Drho, LM_term, C_term, n_tot);

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

   // cudaMemcpy(rho_tot, dev_rhotot, n_tot*n_tot*sizeof(cuDoubleComplex),
   // cudaMemcpyDeviceToHost);

   cudaFree(dev_vec);
   cudaFree(dev_aux1);

   return;
}
//##############################################################################
