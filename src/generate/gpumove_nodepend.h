#pragma once

#include "../types.h"
#include "../graph.h"

graph_t gpumove_nodepend(vector<tuple<int, int, uint64_t> > moves)
{
  // Execute all the moves all at once; there are no dependencies
  vector<uint64_t> offsets;
  auto mem = [&](int loc, uint64_t sz) {
    if(loc >= offsets.size()) {
      offsets.resize(loc+1);
    }
    uint64_t& offset = offsets[loc];
    mem_t ret {
      .offset = offset,
      .size   = sz
    };
    offset += sz;
    return ret;
  };
  auto gpu = make_gpu_loc;

  graph_t g;
  for(auto const& [src, dst, size]: moves) {
    g.insert(sendrecv_t {
      .src = gpu(src),
      .dst = gpu(dst),
      .src_mem = mem(src, size),
      .dst_mem = mem(dst, size)
    }, {});
  }

  return g;
}
