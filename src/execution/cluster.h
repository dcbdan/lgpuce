#pragma once

#include <memory>
#include <thread>

#include "device.h"

#include <cuda_runtime.h>
#include "cublas_v2.h"

using device_ptr_t = std::shared_ptr<device_t>;

struct setting_t {
  int num_cpu_apply;
  int num_cpu_comm;
  int num_gpu_apply;
  int num_gpu_comm;

  setting_t():
    num_cpu_apply(1),
    num_cpu_comm(1),
    num_gpu_apply(1),
    num_gpu_comm(1)
  {}
};

struct cluster_t {
  cluster_t() = delete;

  static cluster_t from_graphs(vector<graph_t> const& graphs) {
    int num_cpus = 0;
    int num_gpus = 0;
    for(auto const& graph: graphs) {
      num_cpus = std::max(num_cpus, graph.num_cpus());
      num_gpus = std::max(num_gpus, graph.num_gpus());
    }

    vector<uint64_t> cpu_mem_sizes(num_cpus, 0);
    vector<uint64_t> gpu_mem_sizes(num_gpus, 0);

    for(auto const& graph: graphs) {
      for(int i = 0; i != cpu_mem_sizes.size(); ++i) {
        cpu_mem_sizes[i] = std::max(cpu_mem_sizes[i], graph.memory_size(make_cpu_loc(i)));
      }

      for(int i = 0; i != gpu_mem_sizes.size(); ++i) {
        gpu_mem_sizes[i] = std::max(gpu_mem_sizes[i], graph.memory_size(make_gpu_loc(i)));
      }
    }

    return cluster_t(cpu_mem_sizes, gpu_mem_sizes);
  }
  static cluster_t from_graph(graph_t const& graph) {
    return from_graphs({graph});
  }

  cluster_t(vector<uint64_t> cpu_mem_sizes, vector<uint64_t> gpu_mem_sizes)
  {
    auto num_cpus = cpu_mem_sizes.size();
    auto num_gpus = gpu_mem_sizes.size();

    if(num_gpus > cuda_get_device_count()) {
      throw std::runtime_error("More gpus used in graph than available");
    }
    if(num_gpus != cuda_get_device_count()) {
      std::cout << "Warning: not all gpus are being used" << std::endl;
    }
    if(num_cpus > 1) {
      std::cout << "Warning: more than one cpu device is being used" << std::endl;
    }

    cpu_devices.reserve(num_cpus);
    for(int i = 0; i != num_cpus; ++i) {
      loc_t loc { .device_type = device_type_t::cpu, .id = i };
      cpu_devices.emplace_back(new device_t(this, cpu_mem_sizes[i], loc));
    }

    if(num_gpus > 0) {
      if(cublasCreate(&gpu_handle) != CUBLAS_STATUS_SUCCESS){
        throw std::runtime_error("gpu handle creation");
      }

      gpu_devices.reserve(num_gpus);
      for(int i = 0; i != num_gpus; ++i) {
        loc_t loc { .device_type = device_type_t::gpu, .id = i };
        gpu_devices.emplace_back(new device_t(this, gpu_mem_sizes[i], loc));
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

  void run(
    graph_t const& g,
    setting_t const& s = setting_t())
  {
    vector<std::thread> ts;

    for(auto device_ptr: cpu_devices) {
      ts.emplace_back([device_ptr, &g](){
        device_ptr->prepare(g);
      });
    }

    for(auto device_ptr: gpu_devices) {
      ts.emplace_back([device_ptr, &g](){
        device_ptr->prepare(g);
      });
    }

    for(std::thread& t: ts) {
      t.join();
    }

    ts = vector<std::thread>();
    auto start = std::chrono::high_resolution_clock::now();

    for(auto device_ptr: cpu_devices) {
      ts.emplace_back([device_ptr,s](){
        device_ptr->run(s.num_cpu_apply, s.num_cpu_comm);
      });
    }

    for(auto device_ptr: gpu_devices) {
      ts.emplace_back([device_ptr,s](){
        device_ptr->run(s.num_gpu_apply, s.num_gpu_comm);
      });
    }

    for(std::thread& t: ts) {
      t.join();
    }

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    std::cout << "Graph execution time: " << 1e-9f * duration.count() << "s" << std::endl;
  }

  void* get_handler(loc_t loc) {
    if(loc.device_type == device_type_t::cpu) {
      return nullptr;
    } else {
      return this->get_gpu_handler();
    }
  }

  void* get_gpu_handler() {
    // Assumption: this only gets used if num gpu devices > 0
    return (void*)(&gpu_handle);
  }

  std::mutex& print_lock() {
    return m_print;
  }

private:
  vector<device_ptr_t> cpu_devices;
  vector<device_ptr_t> gpu_devices;

  cublasHandle_t gpu_handle;

  std::mutex m_print;
};

device_t& device_t::get_device_at(loc_t const& loc) {
  return manager->get(loc);
}
void* device_t::get_handler() {
  return manager->get_handler(this_loc);
}
std::mutex& device_t::print_lock() {
  return manager->print_lock();
}
