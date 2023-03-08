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

void main05() {
  int which_gpu = 2;
  cuda_set_device(which_gpu);
  handler_t gpu_handle(make_gpu_loc(which_gpu));

  uint64_t workspace_size = 8 * 1048576;
  void* workspace;
  cudaMalloc(&workspace, workspace_size);
  cublasSetWorkspace(gpu_handle.gpu_handle, workspace, workspace_size);

  uint64_t ni = 10000;
  uint64_t nm = 20;

  void* memory;
  uint64_t mat_size = sizeof(float)*ni*ni;
  uint64_t memory_size = mat_size*(nm + 2);
  cudaMalloc(&memory, memory_size);

  kernel_t init_op = gen_constant({ni,ni}, 1.0);
  kernel_t op = gen_gpu_matmul(ni,ni,ni);
  kernel_t dummy = gen_gpu_matmul(3,3,3);

  vector<char> x(mat_size);
  init_op((void*)nullptr, vector<void*>{}, vector<void*>{(void*)x.data()});

  auto memory_ = [&](int i) {
    return (void*)((char*)memory + i*mat_size);
  };

  cudaMemcpy(
    memory_(0),
    (void*)x.data(),
    mat_size,
    cudaMemcpyHostToDevice );
  cudaMemcpy(
    memory_(1),
    (void*)x.data(),
    mat_size,
    cudaMemcpyHostToDevice );
  cudaDeviceSynchronize();
  //dummy(gpu_handle(), {memory_(0), memory_(1)}, {memory_(2)});

  auto now = std::chrono::high_resolution_clock::now;
  time_point_t base = now();
  vector<tuple<time_point_t, time_point_t>> ts;
  for(int i = 0; i != nm; ++i) {
    time_point_t start = now();
    op(gpu_handle(), {memory_(0), memory_(1)}, {memory_(2+i)});
    cudaDeviceSynchronize();
    time_point_t stop = now();
    ts.emplace_back(start, stop);
  }

  auto fix = [&base](time_point_t const& t) {
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(t - base);
    return duration.count();
  };
  for(auto const& [b,e]: ts) {
    std::cout << fix(b) << "," << fix(e) << "," << "gpu" << std::endl;
  }

  cudaFree(memory);
  cudaFree(workspace);
}

int main() {
  main04();
}
