#pragma once

#include "../types.h"
#include "../misc.h"
#include <cstdint>

kernel_t gen_add(vector<uint64_t> shape) {
  uint64_t total = uint64_product(shape);
  kernel_t ret;
  ret.capacity = 1;
  ret.flops = total;
  ret.op = [total](void*, vector<void*> const& inns, vector<void*> const& outs) {
    float* lhs = (float*)inns[0];
    float* rhs = (float*)inns[1];
    float* out = (float*)outs[0];
    for(uint64_t i = 0; i != total; ++i) {
      out[i] = lhs[i] + rhs[i];
    }
  };
  return ret;
}

kernel_t gen_aggregate(int num_inputs, uint64_t total) {
  kernel_t ret;
  ret.capacity = 1;
  ret.flops = total * (num_inputs + 1);
  ret.op = [num_inputs, total](void*, vector<void*> const& inns, vector<void*> const& outs) {
    float* out = (float*)outs[0];
    std::fill(out, out + total, 0.0);

    for(int which = 0; which != num_inputs; ++which) {
      float* inn = (float*)inns[which];
      for(int i = 0; i != total; ++i) {
        out[i] += inn[i];
      }
    }
  };
  return ret;
}
// add two matrices of size MxN
kernel_t gen_gpu_matadd(
  bool trans_lhs, uint64_t lhs_n_row, uint64_t lhs_n_col,
  bool trans_rhs, uint64_t rhs_n_row, uint64_t rhs_n_col
  )
{
  float alpha = 1;
  float beta = 1;

  cublasOperation_t trans_lhs_ = trans_lhs ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t trans_rhs_ = trans_rhs ? CUBLAS_OP_T : CUBLAS_OP_N;

  uint64_t lhs_M = trans_lhs ? lhs_n_row : lhs_n_col ;
  uint64_t lhs_N = trans_lhs ? lhs_n_col : lhs_n_row ;

  uint64_t rhs_M = trans_rhs ? rhs_n_row : rhs_n_col ;
  uint64_t rhs_N = trans_rhs ? rhs_n_col : rhs_n_row ;

  if(lhs_M != rhs_M || lhs_N != rhs_N) {
    throw std::runtime_error("gen_gpu_matadd: matrix dimensions do not match");
  }

  uint64_t lda = lhs_n_row;
  uint64_t ldb = rhs_n_row;
  uint64_t ldc = lhs_n_row;

  kernel_t ret;
  ret.capacity = 1000;
  ret.flops = lhs_M * lhs_N;
  ret.op = [trans_lhs_,trans_rhs_,lhs_M,lhs_N,alpha,lda,ldb,beta,ldc]
    (void* info, vector<void*> const& inns, vector<void*> const& outs)
  {
    cublasHandle_t* handle = (cublasHandle_t*)info;
    float* data_lhs = (float*)inns[0];
    float* data_rhs = (float*)inns[1];
    float* data_out = (float*)outs[0];
    
    cublasSgeam(*handle, trans_lhs_, trans_rhs_, lhs_M, lhs_N, &alpha,
                data_lhs, lda, &beta, data_rhs, ldb, data_out, ldc);
  };

  return ret;
}

kernel_t gen_gpu_matadd(uint64_t ni, uint64_t nj) {
  return gen_gpu_matadd(false, ni, nj, false, ni, nj);
}