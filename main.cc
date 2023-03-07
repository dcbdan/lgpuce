#include "src/types.h"
#include "src/kernels.h"
#include "src/graph.h"
#include "src/generate/sloppy_matmul.h"
#include "src/generate/hello_gpumatmul.h"
#include "src/generate/hello_3gpu.h"
#include "src/execution/cluster.h"

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

int main() {
  graph_t g = hello_3gpu(4);
  cluster_t manager = cluster_t::from_graph(g);
  manager.run(g);
  manager.run(g);
  manager.run(g);
}

