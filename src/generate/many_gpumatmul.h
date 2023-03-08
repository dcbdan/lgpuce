#pragma once

#include "../types.h"
#include "../graph.h"
#include "../kernels.h"
#include "../kernels/print.h"

tuple<graph_t, graph_t> many_gpumatmul(
  uint64_t ni,
  int num_gpus,
  int num_slots_per_gpu,
  int num_matmul_per_slot)
{
  // 1. create two tensors on every gpu
  // 2. on each gpu,
  //      multiply those two tensors into num_slots_per_gpu outputs
  //      and do that num_matmul_per_slot times
  uint64_t sz = sizeof(float)*ni*ni;

  auto mem = [&sz](int i) {
    return mem_t { .offset = i*sz, .size = sz };
  };

  graph_t init;
  {
    ident_t x = init.insert(apply_t{
      .loc = make_cpu_loc(0),
      .read_mems = {},
      .write_mems = {mem(0)},
      .op = gen_constant({ni,ni}, 1.0)
      }, {});
    ident_t y = init.insert(apply_t{
      .loc = make_cpu_loc(0),
      .read_mems = {},
      .write_mems = {mem(1)},
      .op = gen_constant({ni,ni}, 1.0)
      }, {});

    for(int i = 0; i != num_gpus; ++i) {
      init.insert(sendrecv_t {
        .src = make_cpu_loc(0),
        .dst = make_gpu_loc(i),
        .src_mem = mem(0),
        .dst_mem = mem(0)
        }, {x});
      init.insert(sendrecv_t {
        .src = make_cpu_loc(0),
        .dst = make_gpu_loc(i),
        .src_mem = mem(1),
        .dst_mem = mem(1)
        }, {y});
    }
  }

  graph_t g;

  for(int i = 0; i != num_gpus; ++i) {
    // initialize the slots
    vector<ident_t> sls(num_slots_per_gpu);
    for(int s = 0; s != num_slots_per_gpu; ++s) {
      sls[s] = g.insert(apply_t {
        .loc = make_gpu_loc(i),
        .read_mems = {mem(0), mem(1)},
        .write_mems = {mem(s+2)},
        .op = gen_gpu_matmul(ni,ni,ni) }, {});
    }

    // do the rest of the mamtuls per slot
    for(int m = 1; m < num_matmul_per_slot; ++m) {
      for(int s = 0; s != num_slots_per_gpu; ++s) {
        sls[s] = g.insert(apply_t {
          .loc = make_gpu_loc(i),
          .read_mems = {mem(0), mem(1)},
          .write_mems = {mem(s+2)},
          .op = gen_gpu_matmul(ni,ni,ni) },
        {sls[s]});
      }
    }
  }

  return {init, g};
}
