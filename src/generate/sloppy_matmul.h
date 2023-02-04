#pragma once

#include "../types.h"
#include "tidmanager.h"
#include "../kernels.h"

// ik,kj->ij
graph_t sloppy_matmul(
  uint64_t bi, uint64_t bj, uint64_t bk,
  uint64_t nd,
  int num_devices)
{
  using tid_t = tidmanager_t::tid_t;
  if(bj > 1) {
    throw std::runtime_error("aggregation not implemented yet");
  }

  // Make some devices (managed by tidmanager),
  // but explicitly make sure they can hold enough memory
  uint64_t mat_size = sizeof(float)*nd*nd;
  uint64_t device_size = mat_size*(bi*bk + bj*bk + bi*bj*bk);
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

  if(bj > 1) {
    // TODO the aggregation
  }

  return manager.get_graph();
}
