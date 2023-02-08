#pragma once

#include "../types.h"
#include "tidmanager.h"
#include "../kernels.h"
#include "../kernels/print.h"

// ik,kj->ij
tuple<graph_t, vector<memloc_t>> sloppy_matmul(
  uint64_t bi, uint64_t bj, uint64_t bk,
  uint64_t nd,
  int num_devices)
{
  using tid_t = tidmanager_t::tid_t;

  // Make some devices (managed by tidmanager),
  // but explicitly make sure they can hold enough memory
  uint64_t mat_size = sizeof(float)*nd*nd;
  uint64_t device_size = mat_size*(25 * (bi*bk + bj*bk + bi*bj*bk)); // TODO what is the actual value?
  tidmanager_t manager(num_devices, device_size);

  int device = 0;

  auto init = [&](uint64_t bx, uint64_t by) {
    vector<tid_t> ret;
    for(uint64_t x = 0; x != bx; x++) {
    for(uint64_t y = 0; y != by; y++) {
      loc_t loc {
        .device = device_t::cpu,
        .id = device};
      tid_t tid = manager.apply(loc, gen_ones({nd,nd}), {}, mat_size);
      ret.push_back(tid);
      device = (device + 1) % num_devices;
    }}
    return ret;
  };

  auto lhs_tids = init(bi, bk);
  auto rhs_tids = init(bk, bj);

  vector<tid_t> join;
  for(uint64_t i = 0; i != bi; i++) {
  for(uint64_t j = 0; j != bj; j++) {
  for(uint64_t k = 0; k != bk; k++) {
    auto const& lhs = lhs_tids[i*bk + k];
    auto const& rhs = rhs_tids[k*bj + j];
    loc_t loc = manager.get_init_loc(lhs);
    tid_t out = manager.apply(loc, gen_cpu_matmul(nd,nd,nd), {lhs, rhs}, mat_size);
    join.push_back(out);
  }}}

  vector<memloc_t> out;
  vector<tid_t> out_tids;
  if(bj > 1) {
    for(uint64_t i = 0; i != bi; i++) {
    for(uint64_t j = 0; j != bj; j++) {
      vector<tid_t> aggs;
      for(uint64_t k = 0; k != bk; k++) {
        aggs.push_back(join[i*bj*bk + j*bk + k]);
      }
      tid_t out_tid = manager.sloppy_reduce(aggs);
      out.push_back(manager.get_memloc(out_tid));
      out_tids.push_back(out_tid);
    }}
  }

  // Now call print the print kernels and have each output depend on the next
  loc_t loc0 = loc_t{ .device = device_t::cpu, .id = 0 };
  auto print_op = gen_print({nd,nd});
  auto print_cmd = [&](mem_t p_mem) {
    return apply_t {
      .loc = loc0,
      .read_mems = {p_mem},
      .write_mems = {},
      .op = print_op
    };
  };

  graph_t graph = manager.get_graph();

  auto const& [mem, ident] = manager.get_at(out_tids[0], loc0);
  ident_t prev_print = graph.insert(print_cmd(mem), {ident});
  if(out_tids.size() > 1) {
    for(int i = 1; i != out_tids.size(); ++i) {
      auto const& [mem, ident] = manager.get_at(out_tids[i], loc0);
      // Here, prev_ident will make sure they all print one at a time
      prev_print = graph.insert(print_cmd(mem), {ident, prev_print});
    }
  }

  return {graph, out};
}
