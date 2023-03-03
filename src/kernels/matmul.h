#pragma once

#include "../types.h"
#include "../misc.h"

#include <cuda_runtime.h>
#include "cublas_v2.h"

// ik,kj->ij
kernel_t gen_cpu_matmul(uint64_t ni, uint64_t nj, uint64_t nk) {
  return [ni,nj,nk](void**, vector<void*> const& inns, vector<void*> const& outs) {
    float* lhs = (float*)inns[0];
    float* rhs = (float*)inns[1];
    float* out = (float*)outs[0];
    for(uint64_t i = 0; i != ni; ++i) {
    for(uint64_t j = 0; j != nj; ++j) {
      out[i*nj + j] = 0.0;
      for(uint64_t k = 0; k != nk; ++k) {
        out[i*nj + j] += lhs[i*nk + k] * rhs[k*nj + j];
      }
    }}
  };
}

kernel_t gen_gpu_matmul(
  bool trans_lhs, uint64_t lhs_n_row, uint64_t lhs_n_col,
  bool trans_rhs, uint64_t rhs_n_row, uint64_t rhs_n_col,
  float alpha = 1.0f)
{
  float beta = 0.0f;

  cublasOperation_t trans_l = trans_lhs ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t trans_r = trans_rhs ? CUBLAS_OP_T : CUBLAS_OP_N;

  uint64_t I = trans_lhs ? lhs_n_row : lhs_n_col;
  uint64_t J = trans_rhs ? rhs_n_col : rhs_n_row;

  uint64_t KL = trans_lhs ? lhs_n_col : lhs_n_row;
  uint64_t KR = trans_rhs ? rhs_n_row : rhs_n_col;

  if(KL != KR) {
    throw std::runtime_error("Should not happen");
  }

  uint64_t lda = lhs_n_row;
  uint64_t ldb = rhs_n_row;
  uint64_t ldc = I;

  return [trans_l,trans_r,I,J,KL,alpha,lda,ldb,beta,ldc]
    (void** info, vector<void*> const& inns, vector<void*> const& outs)
  {
    cublasHandle_t* handle = (cublasHandle_t*)info[0];
    float* data_lhs = (float*)inns[0];
    float* data_rhs = (float*)inns[1];
    float* data_out = (float*)outs[0];

    cublasSgemm(*handle, trans_l, trans_r, I, J, KL, &alpha,
                data_lhs, lda, data_rhs, ldb, &beta, data_out, ldc);
  };
}

kernel_t gen_gpu_matmul(uint64_t ni, uint64_t nj, uint64_t nk) {
  return gen_gpu_matmul(false, ni, nk, false, nj, nk, 1.0);
}


