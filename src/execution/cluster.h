#pragma once

#include <memory>

#include "device.h"
#include "cpu_device.h"
#include "gpu_device.h"

#include <cuda_runtime.h>
#include "cublas_v2.h"

using device_ptr_t = std::shared_ptr<device_t>;

struct cluster_t {
  cluster_t(graph_t const& g, int num_cpus, int num_gpus):
    graph(g)
  {
    cpu_devices.reserve(num_cpus);
    for(int i = 0; i != num_cpus; ++i) {
      loc_t loc { .device_type = device_type_t::cpu, .id = i };
      cpu_devices.emplace_back(new cpu_device_t(this, graph, loc));
    }

    if(num_gpus > 0) {
      if(cublasCreate(&gpu_handle) != CUBLAS_STATUS_SUCCESS){
        throw std::runtime_error("gpu handle creation");
      }
      if(num_gpus != 1) {
        // TODO
        throw std::runtime_error("multiple gpus not implemented!");
      }

      gpu_devices.reserve(num_gpus);
      for(int i = 0; i != num_gpus; ++i) {
        loc_t loc { .device_type = device_type_t::gpu, .id = i };
        gpu_devices.emplace_back(new gpu_device_t(this, graph, loc));
      }
    }
  }

  ~cluster_t() {
    if(gpu_devices.size() > 0) {
      cublasDestroy(gpu_handle);
    }
  }

  device_t& get(loc_t loc) {
    if(loc.device_type == device_type_t::cpu) {
      return *cpu_devices[loc.id];
    } else {
      return *gpu_devices[loc.id];
    }
  }

  void set_cpu_params() {
    throw std::runtime_error("not implemented");
  }

  void set_gpu_params() {
    throw std::runtime_error("not implemented");
  }

  void run()
  {
    vector<std::thread> ts;

    for(auto device_ptr: cpu_devices) {
      ts.emplace_back([device_ptr](){
        device_ptr->run();
      });
    }

    for(auto device_ptr: gpu_devices) {
      ts.emplace_back([device_ptr](){
        device_ptr->run();
      });
    }

    for(std::thread& t: ts) {
      t.join();
    }
  }

  void* get_gpu_handler() {
    // Assumption: this only gets called if num gpu devices > 0
    return (void*)(&gpu_handle);
  }

private:
  graph_t const& graph;
  vector<device_ptr_t> cpu_devices;
  vector<device_ptr_t> gpu_devices;

  cublasHandle_t gpu_handle;
};

device_t& device_t::get_device_at(loc_t const& loc) {
  return manager->get(loc);
}
