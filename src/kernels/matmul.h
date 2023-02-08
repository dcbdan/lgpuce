#pragma once

#include "../types.h"
#include "../misc.h"

// ik,kj->ij
kernel_t gen_cpu_matmul(uint64_t ni, uint64_t nj, uint64_t nk) {
  return [ni,nj,nk](vector<void*> const& inns, vector<void*> const& outs) {
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


