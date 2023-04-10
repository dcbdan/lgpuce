#pragma once

#include "../types.h"
#include "../misc.h"
#include <cstdint>

kernel_t gen_gpu_move(uint64_t size) {

  kernel_t ret;
  ret.capacity = 1;
  ret.flops = size;
  ret.op = [size](void*, vector<void*> const& inns, vector<void*> const& outs) {
    float* inn = (float*)inns[0];
    float* out = (float*)outs[0];
    
    cudaMemcpy(out, inn, size, cudaMemcpyDeviceToDevice);
  };

  return ret;
}
