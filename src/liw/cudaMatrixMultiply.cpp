#include "liw/cudaMatrixMultiply.h"
#include <iostream>
#include <stdexcept>

#define CHECK_CUDA(call) checkCuda((call), __FILE__, __LINE__)
#define CHECK_CUBLAS(call) checkCublas((call), __FILE__, __LINE__)

CudaMatrixMultiply::CudaMatrixMultiply() {
  CHECK_CUBLAS(cublasCreate(&handle));
}

CudaMatrixMultiply::~CudaMatrixMultiply() {
  CHECK_CUBLAS(cublasDestroy(handle));
}

Eigen::MatrixXd CudaMatrixMultiply::multiply(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B) {
  if (A.cols() != B.rows()) {
    throw std::invalid_argument("Number of columns of A must match number of rows of B");
  }

  int M = A.rows();
  int K = A.cols();
  int N = B.cols();

  Eigen::MatrixXd C(M, N);

  double* d_A;
  double* d_B;
  double* d_C;

  CHECK_CUDA(cudaMalloc((void**)&d_A, M * K * sizeof(double)));
  CHECK_CUDA(cudaMalloc((void**)&d_B, K * N * sizeof(double)));
  CHECK_CUDA(cudaMalloc((void**)&d_C, M * N * sizeof(double)));

  CHECK_CUDA(cudaMemcpy(d_A, A.data(), M * K * sizeof(double), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_B, B.data(), K * N * sizeof(double), cudaMemcpyHostToDevice));

  const double alpha = 1.0;
  const double beta = 0.0;

  CHECK_CUBLAS(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, d_A, M, d_B, K, &beta, d_C, M));

  CHECK_CUDA(cudaMemcpy(C.data(), d_C, M * N * sizeof(double), cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(d_A));
  CHECK_CUDA(cudaFree(d_B));
  CHECK_CUDA(cudaFree(d_C));

  return C;
}

void CudaMatrixMultiply::checkCuda(cudaError_t result, const char* file, int line) {
  if (result != cudaSuccess) {
    std::cerr << "CUDA Error: " << file << ":" << line << ", " << cudaGetErrorString(result) << std::endl;
    std::exit(1);
  }
}

void CudaMatrixMultiply::checkCublas(cublasStatus_t result, const char* file, int line) {
  if (result != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "cuBLAS Error: " << file << ":" << line << ", code: " << result << std::endl;
    std::exit(1);
  }
}
