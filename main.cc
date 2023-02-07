#include "src/types.h"
#include "src/kernels.h"
#include "src/graph.h"
#include "src/generate/sloppy_matmul.h"
#include "src/execution/cpu.h"

int main() {
  //auto [graph, memlocs] = sloppy_matmul(4, 4, 5, 1024, 2);
  //graph.print(std::cout);

  auto [graph, memlocs] = sloppy_matmul(2, 2, 2, 1024, 1);
  graph.print(std::cout);
  loc_t cpu0 { .device = device_t::cpu, .id = 0 };
  devicemanager_t manager(graph, cpu0);
  std::cout << "-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*" << std::endl;
  manager.launch(20);
}
