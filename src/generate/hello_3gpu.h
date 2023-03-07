#pragma once

#include "../types.h"
#include "../graph.h"
#include "../kernels.h"
#include "../kernels/print.h"

graph_t hello_3gpu(uint64_t ni)
{
  // 1. Create ni by ni tensors x and y on cpu
  // 2. Move them to gpu 1
  // 3. compute x * y on gpu 1 and move it to gpu 2
  // 4. compute x * (x * y) on gpu 2 and move it to gpu 3
  // 5. compute x * (x * (x * y) ) on gpu3 and move it to cpu
  // 6. print the result on the cpu
  uint64_t sz = sizeof(float)*ni*ni;

  loc_t cpu  { .device_type = device_type_t::cpu, .id = 0 };
  loc_t gpu1 { .device_type = device_type_t::gpu, .id = 0 };
  loc_t gpu2 { .device_type = device_type_t::gpu, .id = 1 };
  loc_t gpu3 { .device_type = device_type_t::gpu, .id = 2 };

  mem_t mem0 { .offset = 0,    .size = sz };
  mem_t mem1 { .offset = sz,   .size = sz };
  mem_t mem2 { .offset = 2*sz, .size = sz };

  graph_t g;

  ident_t construct_x_cpu = g.insert(
    apply_t{
      .loc = cpu,
      .read_mems = {},
      .write_mems = {mem0},
      .op = gen_constant({ni,ni}, 1.0) },
    {}
  );
  ident_t construct_y_cpu = g.insert(
    apply_t{
      .loc = cpu,
      .read_mems = {},
      .write_mems = {mem1},
      .op = gen_constant({ni,ni}, 1.0) },
    {}
  );


  ident_t move_x_1 = g.insert(sendrecv_t{ .src=cpu, .dst=gpu1, .src_mem=mem0, .dst_mem=mem0}, {construct_x_cpu});
  ident_t move_x_2 = g.insert(sendrecv_t{ .src=cpu, .dst=gpu2, .src_mem=mem0, .dst_mem=mem0}, {construct_x_cpu});
  ident_t move_x_3 = g.insert(sendrecv_t{ .src=cpu, .dst=gpu3, .src_mem=mem0, .dst_mem=mem0}, {construct_x_cpu});

  ident_t move_y_1 = g.insert(sendrecv_t{ .src=cpu, .dst=gpu1, .src_mem=mem1, .dst_mem=mem1}, {construct_y_cpu});

  ident_t y1 = g.insert(
    apply_t {
      .loc = gpu1,
      .read_mems = {mem0, mem1},
      .write_mems = {mem2},
      .op = gen_gpu_matmul(ni,ni,ni) },
    {move_x_1, move_y_1}
  );
  ident_t move_y_2 = g.insert(
    sendrecv_t { .src=gpu1, .dst=gpu2, .src_mem=mem2, .dst_mem=mem1},
    {y1});

  ident_t y2 = g.insert(
    apply_t {
      .loc = gpu2,
      .read_mems = {mem0, mem1},
      .write_mems = {mem2},
      .op = gen_gpu_matmul(ni,ni,ni) },
    {move_x_2, move_y_2});
  ident_t move_y_3 = g.insert(
    sendrecv_t { .src=gpu2, .dst=gpu3, .src_mem=mem2, .dst_mem=mem1},
    {y2});

  ident_t y3 = g.insert(
    apply_t {
      .loc = gpu3,
      .read_mems = {mem0, mem1},
      .write_mems = {mem2},
      .op = gen_gpu_matmul(ni,ni,ni) },
    {move_x_3, move_y_3});
  ident_t move_y_4 = g.insert(
    sendrecv_t { .src=gpu3, .dst=cpu, .src_mem=mem2, .dst_mem=mem0},
    {y3});

  g.insert(
    apply_t{
      .loc = cpu,
      .read_mems = {mem0},
      .write_mems = {},
      .op = gen_print({ni,ni}) },
    {move_y_4}
  );

  return g;
}


