#pragma once

#include "../types.h"
#include "../misc.h"

kernel_t gen_add(vector<uint64_t> shape) {
  uint64_t total = uint64_product(shape);
  return [total](vector<void*> const& inns, vector<void*> const& outs) {
    float* lhs = (float*)inns[0];
    float* rhs = (float*)inns[1];
    float* out = (float*)outs[0];
    for(uint64_t i = 0; i != total; ++i) {
      out[i] = lhs[i] + rhs[i];
    }
  };
}

kernel_t gen_inplace_lhs_add(vector<uint64_t> shape) {
  uint64_t total = uint64_product(shape);
  return [total](vector<void*> const& inns, vector<void*> const& outs) {
    float* lhs = (float*)outs[0];
    float* rhs = (float*)inns[0];
    for(uint64_t i = 0; i != total; ++i) {
      lhs[i] += rhs[i];
    }
  };
}

kernel_t gen_inplace_rhs_add(vector<uint64_t> shape) {
  uint64_t total = uint64_product(shape);
  return [total](vector<void*> const& inns, vector<void*> const& outs) {
    float* lhs = (float*)inns[0];
    float* rhs = (float*)outs[0];
    for(uint64_t i = 0; i != total; ++i) {
      rhs[i] += lhs[i];
    }
  };
}
