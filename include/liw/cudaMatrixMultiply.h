#ifndef CUDA_MATRIX_MULTIPLY_H
#define CUDA_MATRIX_MULTIPLY_H

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <Eigen/Dense>

class CudaMatrixMultiply {
 public:
  CudaMatrixMultiply();
  ~CudaMatrixMultiply();

  Eigen::MatrixXd multiply(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B);

 private:
  cublasHandle_t handle;

  void checkCuda(cudaError_t result, const char* file, int line);
  void checkCublas(cublasStatus_t result, const char* file, int line);
};

#endif  // CUDA_MATRIX_MULTIPLY_H
