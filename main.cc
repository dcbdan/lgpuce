#include "src/types.h"
#include "src/kernels.h"
#include "src/graph.h"
#include "src/generate/sloppy_matmul.h"

int main() {
  graph_t graph = sloppy_matmul(4, 1, 5, 1024, 2);
  graph.print(std::cout);
}
