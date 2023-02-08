#pragma once

#include <iostream> // TODO

#include "../types.h"
#include "../graph.h"
#include "interval_lock.h"

#include <queue>

#include <thread>
#include <mutex>
#include <condition_variable>

// TODO
// - read/write locks and total memory of each device
// - how should sending and recving be accomplished?

struct devicemanager_t {
  devicemanager_t(graph_t const& g, uint64_t memory_size, loc_t this_loc):
    graph(g), counts(g.size()), num_remaining(0), memory(memory_size)
  {
    auto const& info = g.get_info();
    for(int ident = 0; ident != info.size(); ++ident) {
      auto const& [parent, _, children] = info[ident];
      if(is_signal_location(parent, this_loc)) {
        // This command needs to happen here!
        num_remaining++;

        int count = 0;
        for(auto const& child_ident: children) {
          auto const& [child, _0, _1] = info[child_ident];
          if(is_signal_location(child, this_loc)) {
            count++;
          }
        }
        if(count == 0) {
          ready.push(ident);
        } else {
          counts[ident] = count;
        }
      }
    }
  }

  devicemanager_t(graph_t const& g, loc_t this_loc):
    devicemanager_t(g, g.memory_size(this_loc), this_loc)
  {}

  void apply_runner(int runner_id) {
    int which;
    while(true) {
      // Get a command that needs to be executed, or return
      {
        std::unique_lock lk(m);
        cv.wait(lk, [this](){return num_remaining == 0 || ready.size() > 0;});
        if(num_remaining == 0) {
          return;
        }
        which = ready.front();
        ready.pop();
      }

      // Do the command execution
      {
        apply_t const& apply = std::get<apply_t>(graph[which]);
        vector<interval_t> read_intervals, write_intervals;
        vector<void*> read_mems, write_mems;
        char* data = memory.data();
        for(auto const& mem: apply.read_mems) {
          read_intervals.push_back(mem.interval());
          read_mems.push_back(data + mem.offset);
        }
        for(auto const& mem: apply.write_mems) {
          write_intervals.push_back(mem.interval());
          write_mems.push_back(data + mem.offset);
        }
        auto memlock = memory_lock.acquire(read_intervals, write_intervals);
        apply.op(read_mems, write_mems);
        {
          std::unique_lock lk_print(m_print);
          std::cout << "cmd " << which <<
                       " @ runner " << runner_id << ": " <<
                       graph.get_command(which) << std::endl;
        }
      }

      {
        std::unique_lock lk(m);
        num_remaining--;
        auto const& parents = graph.get_parents(which);
        for(auto const& parent : parents) {
          counts[parent]--;
          if(counts[parent] == 0) {
            ready.push(parent);
          }
        }
      }
      cv.notify_all();
    }
  }

  void launch(int num_runners) {
    std::vector<std::thread> runners;
    runners.reserve(num_runners);
    for(int i = 0; i != num_runners; ++i) {
      runners.emplace_back([this, i](){ return this->apply_runner(i); });
    }
    for(std::thread& t: runners) {
      t.join();
    }
  }

private:
  graph_t const& graph;
  vector<int> counts;
  int num_remaining;
  std::queue<int> ready;

  interval_lock_t memory_lock;
  vector<char> memory;

  std::mutex m, m_print; // TODO remove m_print
  std::condition_variable cv;
};
