#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>



int main(int argc, char **argv) {
  int N = (argc > 1) ? atoi(argv[1]) : 1024;
  size_t size = N*N*sizeof(float);

  float *A = (float *)malloc(size);
  float *B = (float *)malloc(size);
  float *C = (float *)malloc(size);

  for (int i = 0; i < N*N; i++) {
    A[i] = rand() % 100 / 100.0f;
    B[i] = rand() % 100 / 100.0f;
  }

  float *d_A, *d_B, *d_C;
  cudaMalloc((void**)&d_A, size);
  cudaMalloc((void**)&d_B, size);
  cudaMalloc((void**)&d_C, size);

  cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);


  cublasHandle_t handle;
  cublasCreate(&handle);

  float alpha = 1.0f;
  float beta = 0.0f;

  cublasSgemm(handle,
    CUBLAS_OP_N, CUBLAS_OP_N,
    N, N, N,
    &alpha,
    d_B, N,
    d_A, N,
    &beta,
    d_C, N);
  cudaDeviceSynchronize();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start); 

  cublasSgemm(handle,
    CUBLAS_OP_N, CUBLAS_OP_N,
    N, N, N,
    &alpha,
    d_B, N,
    d_A, N,
    &beta,
    d_C, N);

  cudaEventRecord(stop); 
  cudaEventSynchronize(stop);


  float milliseconds = 0; 
  cudaEventElapsedTime(&milliseconds, start, stop); 

  cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);


  printf("cuBLAS execution time (N=%d): %f seconds\n", N, milliseconds/1000.0f);

  cublasDestroy(handle);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  free(A);
  free(B);
  free(C);

  return 0;
}
