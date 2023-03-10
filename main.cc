#include "src/types.h"
#include "src/kernels.h"
#include "src/graph.h"
#include "src/generate/sloppy_matmul.h"
#include "src/generate/hello_gpumatmul.h"
#include "src/generate/hello_3gpu.h"
#include "src/generate/many_gpumatmul.h"
#include "src/execution/cluster.h"
#include "src/generate/hello_gpumove.h"
#include "src/generate/gpumove_nodepend.h"

#include <sstream>
#include <cstdlib>

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
  graph_t g = hello_3gpu(10000);
  cluster_t manager = cluster_t::from_graph(g);
  manager.run(g);
  manager.log_time_events("hello_3gpu.log");
}

void main04() {
  uint64_t ni = 4000;
  int num_gpus = 3;
  int num_slots_per_gpu = 8;
  int num_matmul_per_slot = 25;

  auto [init, g] = many_gpumatmul(ni, num_gpus, num_slots_per_gpu, num_matmul_per_slot);

  cluster_t manager = cluster_t::from_graphs({init,g});
  manager.run(init);

  setting_t s;
  s.num_gpu_apply = 1;
  manager.run(g, s);

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
  handler_t gpu_handle(make_gpu_loc(which_gpu), true);

  uint64_t ni = 1000;
  uint64_t nm = 40;

  void* memory;
  uint64_t mat_size = sizeof(float)*ni*ni;
  uint64_t memory_size = mat_size*(2*nm + 2);
  cudaMalloc(&memory, memory_size);

  kernel_t init_op = gen_constant({ni,ni}, 1.0);
  kernel_t op = gen_gpu_matmul(ni,ni,ni);

  vector<char> x(mat_size);
  init_op((void*)nullptr, vector<void*>{}, vector<void*>{(void*)x.data()});

  auto memory_ = [&](int i) {
    return (void*)((char*)memory + i*mat_size);
  };

  cudaStream_t stream1;
  cudaStreamCreate(&stream1);

  cudaStream_t stream2;
  cudaStreamCreate(&stream2);

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

  auto now = std::chrono::high_resolution_clock::now;
  time_point_t base = now();
  vector<tuple<time_point_t, time_point_t, int>> ts;
  for(int i = 0; i != nm; ++i) {
    time_point_t start1 = now();
    cublasSetStream(gpu_handle.gpu_handle, stream1);
    op(gpu_handle(), {memory_(0), memory_(1)}, {memory_(2+2*i)});

    time_point_t start2 = now();
    cublasSetStream(gpu_handle.gpu_handle, stream2);
    op(gpu_handle(), {memory_(0), memory_(1)}, {memory_(2+2*i + 1)});

    cudaStreamSynchronize(stream1);
    time_point_t stop1 = now();

    cudaStreamSynchronize(stream2);
    time_point_t stop2 = now();

    ts.emplace_back(start1, stop1, 1);
    ts.emplace_back(start2, stop2, 2);
  }

  auto fix = [&base](time_point_t const& t) {
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(t - base);
    return duration.count();
  };
  for(auto const& [b,e,i]: ts) {
    std::cout << fix(b) << "," << fix(e) << "," << "s" << i << std::endl;
  }

  cudaFree(memory);
  cudaStreamDestroy(stream1);
  cudaStreamDestroy(stream2);
}

void main06(int argc, char** argv)
{
  if(argc < 3) {
    throw std::runtime_error("Must supply arguments");
  }

  uint64_t size = sizeof(float)*10000*10000;
  int n_gpu  = 3;
  int n_blob = atoi(argv[1]);
  int n_move = atoi(argv[2]);
  setting_t s;
  if(argc >= 4) {
    s.num_gpu_comm = atoi(argv[3]);
  }

  std::cout << "n_blob  " << n_blob << std::endl;
  std::cout << "n_move  " << n_move << std::endl;
  std::cout << "n_comm  " << s.num_gpu_comm << std::endl;

  graph_t g = hello_gpumove(size, n_gpu, n_blob, n_move);

  cluster_t manager = cluster_t::from_graph(g);

  manager.run(g, s);

  std::stringstream ss;
  ss << "hello_gpumove" << n_gpu
     << "_sz" << size
     << "_bl" << n_blob
     << "_mv" << n_move
     << ".log";
  //manager.log_time_events(ss.str());
  manager.log_time_events("move.log");
}

void main07() {
  uint64_t GiB = 1 << 30;
  uint64_t sz = 6*GiB;
  std::cout << sz << std::endl;
  void* memory0;
  void* memory1;

  // Note: if you don't enable peer access, nvlink isn't used

  cudaSetDevice(0);
  cudaDeviceEnablePeerAccess(1, 0);
  cudaMalloc(&memory0, sz);
  std::cout << __LINE__ << std::endl;

  cudaSetDevice(1);
  cudaDeviceEnablePeerAccess(0, 0);
  cudaMalloc(&memory1, sz);
  std::cout << __LINE__ << std::endl;

  cudaMemcpy(memory0, memory1, sz, cudaMemcpyDefault);
  std::cout << __LINE__ << std::endl;

  cudaDeviceSynchronize();
  std::cout << __LINE__ << std::endl;

  cudaSetDevice(0);
  cudaFree(memory0);
  std::cout << __LINE__ << std::endl;

  cudaSetDevice(1);
  cudaFree(memory1);
  std::cout << __LINE__ << std::endl;
}

void main08(int argc, char** argv) {
  using namespace std::chrono_literals;

  if(argc < 1) {
    throw std::runtime_error("Must supply num_gpu_comm");
  }

  setting_t s;
  s.num_cpu_apply = 0;
  s.num_cpu_comm  = 0;
  s.num_gpu_apply = 0;
  s.num_gpu_comm  = atoi(argv[1]);
  std::cout << "num_gpu_comm " << s.num_gpu_comm << std::endl;

  uint64_t GiB = 1 << 30;
  uint64_t sz = 1*GiB;

  auto gen_graph = [sz](int a, int b, int n) {
    vector<tuple<int, int, uint64_t>> moves;
    for(int i = 0; i != n; ++i) {
      moves.emplace_back(a, b, sz);
      moves.emplace_back(b, a, sz);
    }
    return gpumove_nodepend(moves);
  };

  int n01 = 1;
  int n02 = 1;
  int n12 = 2;
  vector<graph_t> gs { gen_graph(0, 1, n01), gen_graph(0, 2, n02), gen_graph(1, 2, n12) };
  cluster_t manager = cluster_t::from_graphs(gs);

  for(auto const& g : gs) {
    manager.run(g, s);
  }
}

void main09() {
  uint64_t GiB = 1 << 30;

  // Enable gpu nvlink
  int num_gpus = 3;
  for(int i = 0; i != num_gpus; ++i) {
  for(int j = 0; j != num_gpus; ++j) {
    if(i != j) {
      cuda_device_enable_peer_access(i,j);
    }
  }}

  int d0 = 1;
  int d1 = 2;

  // Target: get overlap in send and recv
  cudaSetDevice(d0);
  void* m0_a;
  void* m0_b;
  cudaMalloc(&m0_a, GiB);
  cudaMalloc(&m0_b, GiB);

  cudaSetDevice(d1);
  void* m1_a;
  void* m1_b;
  cudaMalloc(&m1_a, GiB);
  cudaMalloc(&m1_b, GiB);

  cudaSetDevice(d0);
  cudaDeviceSynchronize();
  cudaSetDevice(d1);
  cudaDeviceSynchronize();

  for(int i = 0; i != 2; ++i) {
    auto now = std::chrono::high_resolution_clock::now;
    time_point_t start = now();
    cudaSetDevice(d1);
    cudaMemcpyAsync(m1_a, m0_a, GiB, cudaMemcpyDefault, cudaStreamPerThread);
    cudaSetDevice(d0);
    cudaMemcpyAsync(m0_b, m1_b, GiB, cudaMemcpyDefault, cudaStreamPerThread);

    cudaSetDevice(d0);
    cudaDeviceSynchronize();
    cudaSetDevice(d1);
    cudaDeviceSynchronize();

    time_point_t stop = now();

    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    std::cout << "Execution time: " << 1e-9f * duration.count() << "s" << std::endl;
  }

}

int main(int argc, char** argv){
  //main06(argc, argv);
  //main03();
  //main02();
  //main07();
  main08(argc, argv);
  //main09();
}
