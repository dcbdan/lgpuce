#pragma once

#include "../types.h"
#include "../graph.h"

#include <thread>

struct cluster_t;

struct device_t {
  device_t(cluster_t* manager, graph_t const& g):
    manager(manager), graph(g)
  {}

  virtual void run() = 0;

  device_t& get_device_at(loc_t const& loc);

  cluster_t* manager;
  graph_t const& graph;
};
