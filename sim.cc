#include "src/simulate/cluster.h"
#include "src/simulate/sim.h"

#include "src/types.h"
#include "src/kernels/add.h"
#include "src/kernels/constant.h"
#include "src/kernels/print.h"

#include "src/generate/hello_gpumove.h"

using namespace sim;

cluster_t anton_j0() {
  // nvidia tesla p100 9.3 Teraflops single precision
  uint64_t giga = 1e9;
  uint64_t tera = 1e12;
  uint64_t nvidia_tesla_p100 = (tera * 93) / 10;

  auto make_gpu_dev = [&](int i) {
    loc_t gpu = make_gpu_loc(i);
    // Just give it a capacity of 1000
    return device_t {
      .loc = gpu,
      .compute = nvidia_tesla_p100 / 1000,
      .capacity = 1000 };
  };

  auto make_nvlink = [&](int src, int dst) {
    return connection_t {
      .bandwidth = 20 * giga,
      .src = make_gpu_loc(src),
      .dst = make_gpu_loc(dst)
    };
  };

  auto make_cpu_to_gpu = [&](int gpu) {
    return connection_t {
      .bandwidth = 5 * giga, // a guestimate
      .src = make_cpu_loc(0),
      .dst = make_gpu_loc(gpu)
    };
  };

  auto make_gpu_to_cpu = [&](int gpu) {
    return connection_t {
      .bandwidth = 5 * giga, // a guestimate
      .src = make_gpu_loc(gpu),
      .dst = make_cpu_loc(0)
    };
  };

  cluster_t cluster;
  cluster.insert_device(device_t {
    .loc      = make_cpu_loc(0),
    .compute  = 5 * giga, // just a guesstimate per core
    .capacity = 24
  });

  cluster.insert_device(make_gpu_dev(0));
  cluster.insert_device(make_gpu_dev(1));
  cluster.insert_device(make_gpu_dev(2));

  cluster.insert_connection(make_nvlink(0, 1));
  cluster.insert_connection(make_nvlink(1, 0));

  cluster.insert_connection(make_nvlink(0, 2));
  cluster.insert_connection(make_nvlink(2, 0));

  cluster.insert_connection(make_nvlink(1, 2));
  cluster.insert_connection(make_nvlink(1, 2));
  cluster.insert_connection(make_nvlink(2, 1));
  cluster.insert_connection(make_nvlink(2, 1));

  cluster.insert_connection(make_cpu_to_gpu(0));
  cluster.insert_connection(make_cpu_to_gpu(1));
  cluster.insert_connection(make_cpu_to_gpu(2));

  cluster.insert_connection(make_gpu_to_cpu(0));
  cluster.insert_connection(make_gpu_to_cpu(1));
  cluster.insert_connection(make_gpu_to_cpu(2));

  return cluster;
}

int main() {
  sim::cluster_t cluster = anton_j0();

  uint64_t size = sizeof(float)*10000*10000;
  int n_gpu  = 3;
  int n_blob = 1;
  int n_move = 10;
  graph_t graph = hello_gpumove(size, n_gpu, n_blob, n_move);

  std::cout << simulate(cluster, graph) << std::endl;
}
