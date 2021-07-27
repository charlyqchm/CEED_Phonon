#include "cuda_subs.h"

cuDoubleComplex  *dev_rhoelec;
cuDoubleComplex  *dev_rhophon;
cuDoubleComplex  *dev_rhotot;
cuDoubleComplex  *dev_rhonew;
cuDoubleComplex  *dev_Htot1;
cuDoubleComplex  *dev_Htot2;
cuDoubleComplex  *dev_Htot3;
cuDoubleComplex  *dev_mutot;
double           *dev_vbath;
double           *dev_fb;
UNINT             Ncores;

//##############################################################################
__global__ void update_H_tot(cuDoubleComplex *H_out, cuDoubleComplex *H_tot,
                             cuDoubleComplex *mu_tot,
                             double *v_bath_mat, double *fb_vec,
                             double sum_xi, double Efield, int n_el,
                             int n_phon, int np_levels, int n_tot){

   cuDoubleComplex aux1;
   cuDoubleComplex aux2;
   cuDoubleComplex aux3;

   int ind  = threadIdx.x + blockIdx.x * blockDim.x;
   int dim2 = n_tot * n_tot;
   int i1   = ind / n_tot;
   int i_e  = i1 / (n_phon*np_levels);

   if (ind < dim2){
      if ( ind == i1 + i1*n_tot ){
         aux1       = make_cuDoubleComplex(fb_vec[i_e] * sum_xi, 0.0e0);
         H_out[ind] = cuCadd(H_tot[ind],aux1);
      }

      aux1 = make_cuDoubleComplex(Efield, 0.0e0);
      aux1 = cuCmul(aux1, mu_tot[ind]);

      aux2 = make_cuDoubleComplex(sum_xi*v_bath_mat[ind], 0.0e0);

      aux3 = cuCadd(aux1,aux2);
      H_out[ind] = cuCadd(H_out[ind], aux3);
   }

   return;
}
//##############################################################################
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
__global__ void build_Hceed(cuDoubleComplex *H_out, cuDoubleComplex *H_in,
                            cuDoubleComplex *H_ceed, double dmu2,
                            double a_ceed, int n_tot){

   int ind  = threadIdx.x + blockIdx.x * blockDim.x;
   int dim2 = n_tot * n_tot;

   if (ind < dim2){
      cuDoubleComplex aux1 = make_cuDoubleComplex(0.0e0, a_ceed*dmu2);
      cuDoubleComplex aux2;
      aux2       = cuCmul(aux1,H_ceed[ind]);
      H_out[ind] = cuCadd(H_in[ind], aux2);
   }

   return;
}
//##############################################################################
void init_cuda(complex<double> *H_tot, complex<double> *mu_tot,
               double *v_bath_mat, double *fb_vec, complex<double> *rho_tot,
               UNINT n_el, UNINT n_phon, UNINT n_tot){

   double gaux = (double) (n_tot*n_tot);
   double taux = (double) Nthreads;

   Ncores = (UNINT) ceil(gaux/taux);

   cudaMalloc((void**) &dev_Htot1  , n_tot*n_tot*sizeof(cuDoubleComplex));
   cudaMalloc((void**) &dev_Htot2  , n_tot*n_tot*sizeof(cuDoubleComplex));
   cudaMalloc((void**) &dev_Htot3  , n_tot*n_tot*sizeof(cuDoubleComplex));
   cudaMalloc((void**) &dev_mutot  , n_tot*n_tot*sizeof(cuDoubleComplex));
   cudaMalloc((void**) &dev_rhotot , n_tot*n_tot*sizeof(cuDoubleComplex));
   cudaMalloc((void**) &dev_rhonew , n_tot*n_tot*sizeof(cuDoubleComplex));
   cudaMalloc((void**) &dev_vbath  , n_tot*n_tot*sizeof(double));
   cudaMalloc((void**) &dev_fb     , n_el*sizeof(double));

   cudaMemcpy(dev_Htot1, H_tot, n_tot*n_tot*sizeof(cuDoubleComplex),
              cudaMemcpyHostToDevice);
   cudaMemcpy(dev_Htot2, H_tot, n_tot*n_tot*sizeof(cuDoubleComplex),
              cudaMemcpyHostToDevice);
   cudaMemcpy(dev_Htot3, H_tot, n_tot*n_tot*sizeof(cuDoubleComplex),
              cudaMemcpyHostToDevice);
   cudaMemcpy(dev_mutot, mu_tot, n_tot*n_tot*sizeof(cuDoubleComplex),
              cudaMemcpyHostToDevice);
   cudaMemcpy(dev_rhotot, rho_tot, n_tot*n_tot*sizeof(cuDoubleComplex),
              cudaMemcpyHostToDevice);
   cudaMemcpy(dev_rhonew, rho_tot, n_tot*n_tot*sizeof(cuDoubleComplex),
              cudaMemcpyHostToDevice);
   cudaMemcpy(dev_vbath, v_bath_mat, n_tot*n_tot*sizeof(double),
              cudaMemcpyHostToDevice);
   cudaMemcpy(dev_fb, fb_vec, n_el*sizeof(double), cudaMemcpyHostToDevice);

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
   cudaFree(dev_vbath);
   cudaFree(dev_fb);

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
                  cuDoubleComplex *dev_C, int dim){

   const cuDoubleComplex alf  = make_cuDoubleComplex(1.0,0.0);
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
double get_trace_cuda(cuDoubleComplex *dev_A, UNINT dim){

   // cuDoubleComplex aux1= make_cuDoubleComplex(0.0e0, 0.0e0);
   complex<double> aux1;
   double          aux2;
   complex<double> aux_vec[dim];
   cuDoubleComplex *dev_vec;

   cudaMalloc((void**) &dev_vec, dim * sizeof(cuDoubleComplex));

   get_diag<<<Ncores, Nthreads>>>(dev_A, dev_vec, dim);

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
                        cuDoubleComplex *dev_Hceed,
                        cuDoubleComplex *dev_mu, cuDoubleComplex *dev_rhoaux,
                        double a_ceed, int n_tot){

   int dim2 = n_tot*n_tot;
   double dmu2;
   cuDoubleComplex *dev_aux1, *dev_aux2;

   cudaMalloc((void**) &dev_aux1, dim2 * sizeof(cuDoubleComplex));
   cudaMalloc((void**) &dev_aux2, dim2 * sizeof(cuDoubleComplex));

   commute_cuda(dev_Hin, dev_mu, dev_Hceed, n_tot);
   commute_cuda(dev_Hceed, dev_Hin, dev_aux1, n_tot);
   matmul_cublas(dev_rhoaux, dev_aux1, dev_aux2, n_tot);

   dmu2 = get_trace_cuda(dev_aux2, n_tot);

   build_Hceed<<<Ncores, Nthreads>>>(dev_Hout, dev_Hin, dev_Hceed, dmu2,
                                     a_ceed, n_tot);

   cudaFree(dev_aux1);
   cudaFree(dev_aux2);

   return;
}
//##############################################################################
void getingmat(complex<double> *matA, cuDoubleComplex *dev_A, int n_tot){

   cudaMemcpy(matA, dev_A, n_tot*n_tot*sizeof(cuDoubleComplex),
              cudaMemcpyDeviceToHost);
   return;
}
//
