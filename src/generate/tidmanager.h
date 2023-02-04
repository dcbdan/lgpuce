#pragma once

#include "../types.h"
#include "memorymanager.h"

// tid == tensor identifier;
// A tid references the a value that is the same across
// locations
struct tidmanager_t {
  using tid_t = uint64_t;

  tidmanager_t(int num_devices, int size_per_device):
    allocators(num_devices, memorymanager_t(size_per_device))
  {}

  tid_t apply(loc_t loc, kernel_t kernel, vector<tid_t> inputs, uint64_t out_size)
  {
    vector<mem_t> mem_inputs;
    vector<ident_t> ident_inputs;
    for(auto tid: inputs) {
      auto [m, i] = this->get_at(tid, loc);
      mem_inputs.push_back(m);
      ident_inputs.push_back(i);
    }

    mem_t mem {
      .offset = allocators[loc.id].allocate(out_size),
      .size = out_size };

    apply_t apply {
      .loc = loc,
      .read_mems = mem_inputs,
      .write_mems = {mem},
      .op = kernel };

    ident_t ident = graph.insert(apply, ident_inputs);
    info.push_back(vector<info_t>());
    info.back().push_back(info_t { .loc = loc, .mem = mem, .id = ident, .size = out_size });

    return info.size() - 1;
  }

  tuple<mem_t, ident_t> get_at(tid_t tid, loc_t loc) {
    for(auto const& [l, m, i, _]: info[tid]) {
      if(l == loc) {
        return {m, i};
      }
    }
    this->move_to(tid, loc);
    auto const& [l, m, i, _] = info[tid].back();
    return {m, i};
  }

  void erase(tid_t tid) {
    for(auto const& [loc, mem, _z, _zz]: info[tid]) {
      allocators[loc.id].deallocate(mem.offset);
    }
    info[tid].resize(0);
    // This could leave info full of holes; oh well:
  }

  graph_t get_graph() {
    return graph;
  }

  loc_t get_init_loc(tid_t tid) const& {
    auto const& [l, _z, _zz, _zzz] = info[tid][0];
    return l;
  }

private:
  graph_t graph;
  vector<memorymanager_t> allocators;

  struct info_t {
    loc_t loc;
    mem_t mem;
    ident_t id;
    uint64_t size;
  };
  // For every tid, store a list of
  //   (which device, at what memory, available after what command)
  vector<vector<info_t>> info;

private:
  void move_to(tid_t tid, loc_t dst_loc) {
    auto const& [src_loc, src_mem, src_ident, size] = info[tid][0];
    mem_t new_mem {
      .offset = allocators[dst_loc.id].allocate(size),
      .size = size };
    sendrecv_t move {
      .src = src_loc,
      .dst = dst_loc,
      .src_mem = src_mem,
      .dst_mem = new_mem };
    command_t cmd = move;
    ident_t new_ident = graph.insert(move, { src_ident });
    info[tid].push_back(info_t {
      .loc  = dst_loc,
      .mem  = new_mem,
      .id   = new_ident,
      .size = size });
  }
};
