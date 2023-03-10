#pragma once

#include "../types.h"
#include "../graph.h"

graph_t hello_gpumove(uint64_t size, int n_gpu, int n_blob, int n_move)
{
  // 1. Create n_blob size blobs on gpu (?)
  // 2. Move each blob to the next gpu n_move times
  // In total, there are
  //   n_blob * n_move
  // moves each of size size
  auto gpu = make_gpu_loc;

  auto mem = [&](int i) {
    return mem_t { .offset = i*size, .size = size };
  };
  auto next = [n_gpu](int i) {
    return (i+1) % n_gpu;
  };

  graph_t g;

  vector<tuple<int, ident_t>> moves(n_blob, {0, -1});
  for(int i = 0; i != n_blob; ++i) {
    auto& [which_gpu, _] = moves[i];
    which_gpu = next(i);
  }

  for(int i = 0; i != n_blob; ++i) {
    auto& [which_gpu, ident] = moves[i];
    auto next_gpu = next(which_gpu);
    ident = g.insert(
      sendrecv_t {
        .src = gpu(which_gpu),
        .dst = gpu(next_gpu),
        .src_mem = mem(i),
        .dst_mem = mem(i) },
      {});
    which_gpu = next_gpu;
  }

  if(n_move > 1) {
    for(int move = 1; move != n_move; ++move) {
      for(int i = 0; i != n_blob; ++i) {
        auto& [which_gpu, ident] = moves[i];
        auto next_gpu = next(which_gpu);
        ident = g.insert(
          sendrecv_t {
            .src = gpu(which_gpu),
            .dst = gpu(next_gpu),
            .src_mem = mem(i),
            .dst_mem = mem(i) },
          {ident});
        which_gpu = next_gpu;
      }
    }
  }

  return g;
}
