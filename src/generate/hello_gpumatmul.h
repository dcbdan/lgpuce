#pragma once

#include "../types.h"
#include "../graph.h"
#include "../kernels.h"
#include "../kernels/print.h"

// ik,kj->ij
graph_t hello_gpumatmul(
  uint64_t ni, uint64_t nj, uint64_t nk)
{
  // 1. Allocate constant input tensors on the cpu
  // 2. Move em to the gpu
  // 3. Matmul on the gpu
  // 4. Move the output to the cpu
  // 5. Print on cpu

  uint64_t lhs_size = sizeof(float)*ni*nk;
  uint64_t rhs_size = sizeof(float)*nk*nj;
  uint64_t out_size = sizeof(float)*ni*nj;

  loc_t cpu { .device_type = device_type_t::cpu };
  loc_t gpu { .device_type = device_type_t::gpu };

  mem_t lhs_mem { .offset = 0,        .size = lhs_size };
  mem_t rhs_mem { .offset = lhs_size, .size = rhs_size };

  mem_t out_cpu_mem { .offset = 0, .size = out_size                   };
  mem_t out_gpu_mem { .offset = lhs_size + rhs_size, .size = out_size };

  graph_t g;

  ident_t construct_lhs = g.insert(
    apply_t{
      .loc = cpu,
      .read_mems = {},
      .write_mems = {lhs_mem},
      .op = gen_constant({ni,nk}, 1.0) },
    {}
  );

  ident_t construct_rhs = g.insert(
    apply_t{
      .loc = cpu,
      .read_mems = {},
      .write_mems = {rhs_mem},
      .op = gen_constant({nk,nj}, 1.0) },
    {}
  );

  ident_t move_lhs = g.insert(
    sendrecv_t{
      .src = cpu,
      .dst = gpu,
      .src_mem = lhs_mem,
      .dst_mem = lhs_mem },
    {construct_lhs}
  );

  ident_t move_rhs = g.insert(
    sendrecv_t{
      .src = cpu,
      .dst = gpu,
      .src_mem = rhs_mem,
      .dst_mem = rhs_mem },
    {construct_rhs}
  );

  ident_t matmul = g.insert(
    apply_t {
      .loc = gpu,
      .read_mems = {lhs_mem, rhs_mem},
      .write_mems = {out_gpu_mem},
      .op = gen_gpu_matmul(ni,nj,nk) },
    {move_lhs, move_rhs}
  );

  ident_t move_out = g.insert(
    sendrecv_t{
      .src = gpu,
      .dst = cpu,
      .src_mem = out_gpu_mem,
      .dst_mem = out_cpu_mem },
    {matmul}
  );

  g.insert(
    apply_t{
      .loc = cpu,
      .read_mems = {out_cpu_mem},
      .write_mems = {},
      .op = gen_print({ni,nj}) },
    {move_out}
  );

  return g;
}


