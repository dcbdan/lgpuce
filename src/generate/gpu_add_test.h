#pragma once

#include "../types.h"
#include "../graph.h"
#include "../kernels.h"
#include "../kernels/print.h"
#include <sys/types.h>

// graph_t gpu_add_test(uint64_t ni)
// {
//   // 1. Create ni by ni tensors x and y on cpu
//   // 2. Move them to gpu 1
//   // 3. compute x * y on gpu 1 and move it to gpu 2
//   // 4. compute x * (x * y) on gpu 2 and move it to gpu 3
//   // 5. compute x * (x * (x * y) ) on gpu3
//   // 6. Move each multiply result to cpu and print
//   uint64_t sz = sizeof(float)*ni*ni;

//   loc_t cpu  { .device_type = device_type_t::cpu, .id = 0 };
//   loc_t gpu1 { .device_type = device_type_t::gpu, .id = 0 };

//   mem_t mem0 { .offset = 0,    .size = sz };
//   mem_t mem1 { .offset = sz,   .size = sz };
//   mem_t mem2 { .offset = 2*sz, .size = sz };

//   graph_t g;

//   ident_t construct_x_cpu = g.insert(apply_t{ .loc = cpu, .read_mems = {}, .write_mems = {mem0}, .op = gen_constant({ni,ni}, 1.0) }, {});
//   ident_t construct_y_cpu = g.insert(apply_t{ .loc = cpu, .read_mems = {}, .write_mems = {mem1}, .op = gen_constant({ni,ni}, 1.0) }, {});

//   ident_t move_x = g.insert(sendrecv_t{ .src=cpu, .dst=gpu1, .src_mem=mem0, .dst_mem=mem0}, {construct_x_cpu});
//   ident_t move_y = g.insert(sendrecv_t{ .src=cpu, .dst=gpu1, .src_mem=mem1, .dst_mem=mem1}, {construct_y_cpu});
//   // ident_t move_test = g.insert(sendrecv_t{ .src=gpu1, .dst=gpu1, .src_mem=mem0, .dst_mem=mem1}, {move_x ,move_y});

//   // write_mem = mem2: allocate new memory for the result; write_mem = mem1: overwrite mem1 with the result
//   ident_t add_x_y = g.insert(apply_t{ .loc = gpu1, .read_mems = {mem0, mem1}, .write_mems = {mem2}, 
//                         .op = gen_gpu_matadd(ni,ni) }, {move_x, move_y});

//   ident_t move_x_cpu = g.insert(sendrecv_t { .src=gpu1, .dst=cpu, .src_mem=mem0, .dst_mem=mem0}, {add_x_y});
//   ident_t move_y_cpu = g.insert(sendrecv_t { .src=gpu1, .dst=cpu, .src_mem=mem1, .dst_mem=mem1}, {add_x_y});
//   ident_t move_z_cpu = g.insert(sendrecv_t { .src=gpu1, .dst=cpu, .src_mem=mem2, .dst_mem=mem2}, {add_x_y});

//   // print a statement saying I'm printing input matrix
//   ident_t print_input_1 = g.insert(apply_t{.loc = cpu, .read_mems = {mem0}, .write_mems = {}, .op = gen_print({ni,ni}) }, {move_x_cpu});
//   ident_t print_input_2 = g.insert(apply_t{.loc = cpu, .read_mems = {mem1}, .write_mems = {}, .op = gen_print({ni,ni}) }, {move_y_cpu});
//   // the output should show at the end
//   g.insert(apply_t{.loc = cpu, .read_mems = {mem2}, .write_mems = {}, .op = gen_print({ni,ni}) }, {print_input_1, print_input_2});

//   return g;
// }

graph_t gpu_add_test(uint64_t ni, uint64_t nj)
{
  // 1. Create ni by ni tensors x and y on cpu
  // 2. Move them to gpu 1
  // 3. compute x * y on gpu 1 and move it to gpu 2
  // 4. compute x * (x * y) on gpu 2 and move it to gpu 3
  // 5. compute x * (x * (x * y) ) on gpu3
  // 6. Move each multiply result to cpu and print
  uint64_t sz = sizeof(float)*ni*nj;

  loc_t cpu  { .device_type = device_type_t::cpu, .id = 0 };
  loc_t gpu1 { .device_type = device_type_t::gpu, .id = 0 };

  mem_t mem0 { .offset = 0,    .size = sz };
  mem_t mem1 { .offset = sz,   .size = sz };
  mem_t mem2 { .offset = 2*sz, .size = sz };

  graph_t g;

  ident_t construct_x_cpu = g.insert(apply_t{ .loc = cpu, .read_mems = {}, .write_mems = {mem0}, .op = gen_constant({ni,nj}, 1.0) }, {});
  ident_t construct_y_cpu = g.insert(apply_t{ .loc = cpu, .read_mems = {}, .write_mems = {mem1}, .op = gen_constant({ni,nj}, 1.0) }, {});

  ident_t move_x = g.insert(sendrecv_t{ .src=cpu, .dst=gpu1, .src_mem=mem0, .dst_mem=mem0}, {construct_x_cpu});
  ident_t move_y = g.insert(sendrecv_t{ .src=cpu, .dst=gpu1, .src_mem=mem1, .dst_mem=mem1}, {construct_y_cpu});
  // ident_t move_test = g.insert(sendrecv_t{ .src=gpu1, .dst=gpu1, .src_mem=mem0, .dst_mem=mem1}, {move_x ,move_y});

  // write_mem = mem2: allocate new memory for the result; write_mem = mem1: overwrite mem1 with the result
  ident_t add_x_y = g.insert(apply_t{ .loc = gpu1, .read_mems = {mem0, mem1}, .write_mems = {mem1}, 
                        .op = gen_gpu_matadd(ni,nj) }, {move_x, move_y});

  ident_t move_x_cpu = g.insert(sendrecv_t { .src=gpu1, .dst=cpu, .src_mem=mem0, .dst_mem=mem0}, {add_x_y});
  ident_t move_y_cpu = g.insert(sendrecv_t { .src=gpu1, .dst=cpu, .src_mem=mem1, .dst_mem=mem1}, {add_x_y});
  // ident_t move_z_cpu = g.insert(sendrecv_t { .src=gpu1, .dst=cpu, .src_mem=mem2, .dst_mem=mem2}, {add_x_y});

  // print a statement saying I'm printing input matrix
  ident_t print_input_1 = g.insert(apply_t{.loc = cpu, .read_mems = {mem0}, .write_mems = {}, .op = gen_print({ni,nj}) }, {move_x_cpu});
  ident_t print_input_2 = g.insert(apply_t{.loc = cpu, .read_mems = {mem1}, .write_mems = {}, .op = gen_print({ni,nj}) }, {print_input_1});
  // ident_t print_input_3 = g.insert(apply_t{.loc = cpu, .read_mems = {mem2}, .write_mems = {}, .op = gen_print({ni,nj}) }, {print_input_2});

  return g;
}