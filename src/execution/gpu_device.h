#pragma once

#include <cuda_runtime.h>
#include "cublas_v2.h"

#include "device.h"

struct gpu_device_t : public device_t {
  gpu_device_t(
    cluster_t* manager,
    graph_t const& g,
    uint64_t memory_size,
    loc_t this_loc):
      device_t(manager, g)
  {
    if(cudaMalloc((void**)&data, memory_size) != cudaSuccess) {
      throw std::runtime_error("cudaMalloc failed");
    }
  }

  ~gpu_device_t() {
    cudaFree(data);
  }

  gpu_device_t(cluster_t* manager, graph_t const& g, loc_t this_loc):
    gpu_device_t(manager, g, g.memory_size(this_loc), this_loc)
  {}

  void run() {
    throw std::runtime_error("not implemented");
  }
private:
  void* data;
};


