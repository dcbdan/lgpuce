#pragma once

#include <queue>
#include <functional> // std::greater

#include "../types.h"
#include "../graph.h"
#include "../execute/interval_chopper.h"

#include "cluster.h"

namespace _simulate_impl {

using std::priority_queue;

template <typename T>
using priority_queue_least = std::priority_queue<T, vector<T>, std::greater<T>>;
// For priority_queue_least, the top most element is the smallest,
// which is the opposite behaviour of priority_queue which puts the
// largest element at the top.

vector<interval_t> to_intervals(vector<mem_t> const& mems) {
  vector<interval_t> ret;
  ret.reserve(mems.size());
  for(auto const& mem: mems) {
    ret.push_back(mem.interval());
  }
  return ret;
}

int command_capacity(command_t const& cmd) {
  if(std::holds_alternative<apply_t>(cmd)) {
    apply_t const& apply = std::get<apply_t>(cmd);
    return apply.op.capacity;
  } else if(std::holds_alternative<sendrecv_t>(cmd)) {
    return 1;
  } else {
  throw std::runtime_error("should not reach");
  return -1;
  }
}

struct _cluster_worker_to_index {
  _cluster_worker_to_index(): n(0) {}

  int operator()(loc_t const& src, loc_t const& dst) {
    tuple<loc_t, loc_t> key{src, dst};

    auto iter = connection_to_worker.find(key);
    if(iter == connection_to_worker.end()) {
      connection_to_worker[key] = n;
      n += 1;
      return n-1;
    }
    return iter->second;
  }

  int operator()(loc_t const& key) {
    auto iter = device_to_worker.find(key);
    if(iter == device_to_worker.end()) {
      device_to_worker[key] = n;
      n += 1;
      return n-1;
    }
    return iter->second;
  }

  unordered_map<tuple<loc_t, loc_t>, int> connection_to_worker;
  unordered_map<loc_t, int> device_to_worker;
  int n;
};

struct sim_manager_t {
  sim_manager_t(sim::cluster_t const& cluster, graph_t const& graph):
    _time(0.0), graph(graph), num_remaining(graph.size()),
    info(graph.size()),
    workers(cluster.num_devices() + cluster.num_connections())
  {
    _cluster_worker_to_index to_index;

    for(ident_t ident = 0; ident != graph.size(); ++ident) {
      command_t const& cmd = graph[ident];
      info_t& i            = info[ident];
      i.num_deps_remaining = graph.num_children(ident);
      i.cost               = cluster.cost_command(cmd);
      i.required_capacity  = command_capacity(cmd);

      if(std::holds_alternative<apply_t>(cmd)) {
        apply_t const& apply = std::get<apply_t>(cmd);
        i.which_worker = to_index(apply.loc);
        // set read and write mems to the correct disjoint regions
        auto [read_mems, write_mems] = disjointify(
          to_intervals(apply.read_mems),
          to_intervals(apply.write_mems));

        i.read_mems.reserve(read_mems.size());
        for(auto const& read_interval : read_mems) {
          i.read_mems.emplace_back(i.which_worker, read_interval);
        }
        i.write_mems.reserve(write_mems.size());
        for(auto const& write_interval : write_mems) {
          i.write_mems.emplace_back(i.which_worker, write_interval);
        }
      } else if(std::holds_alternative<sendrecv_t>(cmd)) {
        sendrecv_t const& move = std::get<sendrecv_t>(cmd);
        i.which_worker = to_index(move.src, move.dst);
        i.read_mems.emplace_back(
          to_index(move.src),
          move.src_mem.interval());
        i.write_mems.emplace_back(
          to_index(move.dst),
          move.dst_mem.interval());
      } else {
        throw std::runtime_error("should not reach");
      }

      if(graph.num_children(ident) == 0) {
        workers[i.which_worker].ready.insert(ident);
      }
    }

    for(auto const& kv : to_index.connection_to_worker) {
      workers[kv.second].current_capacity = 1;
    }
    for(auto const& kv : to_index.device_to_worker) {
      workers[kv.second].current_capacity = cluster.device_capacity(kv.first);
    }

    memory_usage.resize(to_index.n);

    // Get the work started
    fill_with_work();
  }

  // Take a step and return whether or not there is still work
  // being done.
  bool step() {
    auto const [time, op_id, worker_id] = in_progress.top();
    in_progress.pop();

    // save the time and decrement the number remaining
    _time = time;
    num_remaining--;

    auto const& op_info = info[op_id];

    // Now that this op is finished, this worker has this much
    // more capacity
    workers[worker_id].current_capacity += op_info.required_capacity;

    // Don't take up this memory any more
    unlock_memory(op_info.read_mems, op_info.write_mems);

    // Decrement counter for parents and add to ready
    // if possible
    for(auto const& parent_id : graph.get_parents(op_id)) {
      auto& parent_info = info[parent_id];
      parent_info.num_deps_remaining--;
      if(parent_info.num_deps_remaining == 0) {
        workers[parent_info.which_worker].ready.insert(parent_id);
      }
    }

    // At this point, start doing all the work possible;
    fill_with_work();

    // Is there work left?
    return in_progress.size() > 0;
  }

  float current_time() const {
    return _time;
  }

  void verify_all_done() const {
    if(num_remaining != 0) {
      throw std::runtime_error("Did not finish all of the work");
    }
  }

private:
  float _time;

  graph_t const& graph;

  int num_remaining;

  // Store current time, op ident, worker id
  priority_queue_least<tuple<float, ident_t, int>> in_progress;

  // For each op ident, store
  //   number of dependencies remaining
  //   cost in time,
  //   capacity required,
  //   read mem,
  //   write mem,
  //   parents
  using which_mem_t = tuple<int, interval_t>;
  struct info_t {
    int num_deps_remaining;
    float cost;
    int required_capacity;
    int which_worker;
    vector<which_mem_t> read_mems;
    vector<which_mem_t> write_mems;
  };
  vector<info_t> info;

  struct worker_info_t {
    unordered_set<ident_t> ready;
    int current_capacity;
  };
  vector<worker_info_t> workers;

  vector<interval_chopper_t> memory_usage;

private:
  void unlock_memory(
    vector<which_mem_t> const& read_mems,
    vector<which_mem_t> const& write_mems)
  {
    for(auto const& [device, interval]: read_mems) {
      memory_usage[device].decrement(interval);
    }
    for(auto const& [device, interval]: write_mems) {
      memory_usage[device].decrement(interval);
    }
  }

  void lock_memory(
    vector<which_mem_t> const& read_mems,
    vector<which_mem_t> const& write_mems)
  {
    // Assumption: available_memory(read_mems, write_mems) is true
    for(auto const& [device, interval]: read_mems) {
      memory_usage[device].increment(interval);
    }
    for(auto const& [device, interval]: write_mems) {
      memory_usage[device].increment(interval);
    }
  }

  bool available_memory(
    vector<which_mem_t> const& read_mems,
    vector<which_mem_t> const& write_mems)
  {
    // Return whether or not all write mems aren't being used
    for(auto const& [device, interval] : write_mems) {
      if(!memory_usage[device].is_zero(interval)) {
        return false;
      }
    }
    return true;
  }

  void fill_with_work() {
    for(worker_info_t& worker: workers) {
      while(do_more_work(worker)) {}
    }
  }

  // try to schedule more work; return true if work
  // was scheduled and false if no more work is available
  bool do_more_work(worker_info_t& worker) {
    // Invariant: all worker.ready ops have no dependencies remaining
    auto iter = worker.ready.begin();
    for(; iter != worker.ready.end(); ++iter) {
      info_t const& ready_info = info[*iter];
      if(worker.current_capacity >= ready_info.required_capacity &&
         available_memory(ready_info.read_mems, ready_info.write_mems))
      {
        break;
      }
    }

    if(iter == worker.ready.end()) {
      return false;
    }

    ident_t op = *iter;
    worker.ready.erase(iter);
    info_t const& ready_info = info[op];

    in_progress.push({current_time() + ready_info.cost, op, ready_info.which_worker});

    worker.current_capacity -= ready_info.required_capacity;
    lock_memory(ready_info.read_mems, ready_info.write_mems);

    return true;
  }
};

}

// Return the estimate time in seconds to run the graph computation
float simulate(sim::cluster_t const& cluster, graph_t const& graph) {
  if(graph.size() == 0) {
    return 0.0;
  }

  _simulate_impl::sim_manager_t manager(cluster, graph);
  while(manager.step()) {}
  manager.verify_all_done();
  return manager.current_time();
}
