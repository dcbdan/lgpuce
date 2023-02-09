#include "src/types.h"
#include "src/kernels.h"
#include "src/graph.h"
#include "src/generate/sloppy_matmul.h"
#include "src/execution/cpu.h"

int main() {
  int num_devices = 2;
  auto [graph, memlocs] = sloppy_matmul(2, 2, 2, 1024, num_devices);
  graph.print(std::cout);

  std::cout << "-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*" << std::endl;

  cluster_manager_t manager(graph, num_devices);
  manager.run(4, 1);
}
