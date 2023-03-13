#pragma once

#include "../types.h"
#include "../graph.h"
#include "../kernels.h"
#include "../kernels/print.h"

tuple<graph_t, graph_t, graph_t>
inplace_add(uint64_t n_elem, int n_add)
{
  // Add n_add n_elem vectors into a vector of zeros.
  // Way 1:
  uint64_t sz = sizeof(float)*n_elem;

  loc_t cpu = make_cpu_loc(0);

  auto mem = [&sz](int i) {
    return mem_t { .offset = i*sz, .size = sz };
  };

  //////////////////
  graph_t g_init;
  {
    ident_t x = g_init.insert(
      apply_t{ .loc = cpu,
               .read_mems = {},
               .write_mems = {mem(n_add)},
               .op = gen_constant({n_elem}, 0.0) },
      {});
    for(int i = 0; i != n_add; ++i) {
      ident_t y = g_init.insert(
        apply_t{ .loc = cpu,
                 .read_mems = {},
                 .write_mems = {mem(i)},
                 .op = gen_constant({n_elem}, 1.0) },
        {});
    }
  }

  //////////////////
  graph_t g_inplace;
  {
    vector<ident_t> ys;
    for(int i = 0; i != n_add; ++i) {
      ys.push_back(
        g_inplace.insert(
          apply_t { .loc = cpu,
                    .read_mems = {mem(i), mem(n_add)},
                    .write_mems = {mem(n_add)},
                    .op = gen_add({n_elem}) },
          {}));
    }

    g_inplace.insert(apply_t {
                       .loc = cpu,
                       .read_mems = {mem(n_add)},
                       .write_mems = {},
                       .op = gen_print({n_elem}) },
                     ys);
  }

  //////////////////
  graph_t g_at_once;
  {
    vector<mem_t> mems;
    for(int i = 0; i != n_add; ++i) {
      mems.push_back(mem(i));
    }

    ident_t v = g_at_once.insert(
      apply_t { .loc = cpu,
                .read_mems = mems,
                .write_mems = {mem(n_add)},
                .op = gen_aggregate(n_add, n_elem) },
      {});
    g_at_once.insert(apply_t {
                       .loc = cpu,
                       .read_mems = {mem(n_add)},
                       .write_mems = {},
                       .op = gen_print({n_elem}) },
                     {v});

  }

  return {g_init, g_inplace, g_at_once};
}
