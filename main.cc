#include "src/types.h"
#include "src/kernels.h"
#include "src/graph.h"
#include "src/generate/sloppy_matmul.h"
#include "src/generate/hello_gpumatmul.h"
#include "src/generate/hello_3gpu.h"
#include "src/generate/many_gpumatmul.h"
#include "src/execution/cluster.h"

#include <sstream>

void main01() {
  int num_devices = 2;
  auto [graph, memlocs] = sloppy_matmul(2, 2, 2, 1024, num_devices);
  graph.print(std::cout);

  std::cout << "-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*" << std::endl;

  cluster_t manager = cluster_t::from_graph(graph);
  manager.run(graph);
}

void main02() {
  graph_t g = hello_gpumatmul(9998,9999,10000);

  cluster_t manager = cluster_t::from_graph(g);
  manager.run(g);
}

void main03() {
  graph_t g = hello_3gpu(4);
  cluster_t manager = cluster_t::from_graph(g);
  manager.run(g);
  manager.log_time_events("hello_3gpu.log");
}

void main04() {
  uint64_t ni = 10000;
  int num_gpus = 3;
  int num_slots_per_gpu = 4;
  int num_matmul_per_slot = 25;

  auto [init, g] = many_gpumatmul(ni, num_gpus, num_slots_per_gpu, num_matmul_per_slot);

  cluster_t manager = cluster_t::from_graphs({init,g});
  manager.run(init);

  manager.run(g);

  std::stringstream ss;
  ss << "manygpumatmul_ngpu" << num_gpus
     << "_ni" << ni
     << "_slot" << num_slots_per_gpu
     << "_mat" << num_matmul_per_slot
     << ".log";
  manager.log_time_events(ss.str());
}

int main() {
  main04();
}
