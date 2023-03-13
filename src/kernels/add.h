#pragma once

#include "../types.h"
#include "../misc.h"

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
