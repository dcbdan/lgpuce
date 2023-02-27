#pragma once

#include "device.h"

#include "interval_lock.h"

#include <queue>
#include <unordered_map>

#include <mutex>
#include <condition_variable>

struct cpu_device_t: public device_t {
  cpu_device_t() = delete;

  cpu_device_t(
    cluster_t* manager,
    graph_t const& g,
    uint64_t memory_size,
    loc_t this_loc,
    int num_apply_runners = 4):
      device_t(manager, g),
      num_apply_runners(num_apply_runners),
      num_cpucpu_runners(1),
      counts(g.size()), this_loc(this_loc),
      num_remaining(0), memory(memory_size)
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
          set_ready(ident);
        } else {
          counts[ident] = count;
        }
      }
    }
  }

  cpu_device_t(cluster_t* manager, graph_t const& g, loc_t this_loc):
    cpu_device_t(manager, g, g.memory_size(this_loc), this_loc)
  {}

  void apply_runner(int runner_id) {
    int which;
    while(true) {
      // Get a command that needs to be executed, or return
      {
        std::unique_lock lk(m);
        cv.wait(lk, [this](){return num_remaining == 0 || apply_ready.size() > 0;});
        if(num_remaining == 0) {
          return;
        }
        which = apply_ready.front();
        apply_ready.pop();
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
                       " @ apply runner " << runner_id << ": " <<
                       graph.get_command(which) << std::endl;
        }
      }

      // Tell everyone that a command has been executed
      this->completed_command(which);
    }
  }

  void communicate_cpucpu_runner(int runner_id) {
    cc_action_t action;
    ident_t can_recv_ident;
    while(true) {
      std::unique_lock lk(m);
      cv.wait(lk, [this, &action, &can_recv_ident](){
        if(num_remaining == 0) {
          action = cc_action_t::shutdown;
          return true;
        }
        if(cc.recv_do.size() > 0) {
          action = cc_action_t::do_recv;
          return true;
        }
        if(cc.send_start.size() > 0) {
          action = cc_action_t::start_send;
          return true;
        }
        if(cc.send_complete.size() > 0) {
          action = cc_action_t::complete_send;
          return true;
        }
        if(cc.can_send.size() > 0) {
          action = cc_action_t::can_send;
          return true;
        }
        for(auto const& [ready_recv_ident,cnt]: cc.recv_ready) {
          if(cnt == 2) {
            can_recv_ident = ready_recv_ident;
            action = cc_action_t::can_recv;
            return true;
          }
        }

        return false;
      });

      if(action == cc_action_t::shutdown) {
        return;
      } else if(action == cc_action_t::can_send) {
        // notify the recver that we can do a send
        ident_t which = cc.can_send.front();
        cc.can_send.pop();

        lk.unlock();

        sendrecv_t const& move = std::get<sendrecv_t>(graph[which]);
        auto& other_manager = get_cpu_device_at(move.dst);
        other_manager.cc_from_send_to_recv_can_send(which);
      } else if(action == cc_action_t::can_recv) {
        // notify the sender that we can do a recv
        auto& which = can_recv_ident;
        cc.recv_ready.erase(which);

        lk.unlock();

        sendrecv_t const& move = std::get<sendrecv_t>(graph[which]);
        auto& other_manager = get_cpu_device_at(move.src);
        other_manager.cc_from_recv_to_send_can_recv(which);
      } else if(action == cc_action_t::start_send) {
        // as the sender, put a read lock around the memory
        // and give the memory to the recver
        ident_t which = cc.send_start.front();
        cc.send_start.pop();

        lk.unlock();

        sendrecv_t const& move = std::get<sendrecv_t>(graph[which]);

        interval_t send_interval = move.src_mem.interval();
        char* send_ptr = memory.data() + move.src_mem.offset;
        memory_lock.lock({send_interval}, {});

        auto& other_manager = get_cpu_device_at(move.dst);
        other_manager.cc_from_send_to_recv_started_send(which, send_ptr);
      } else if(action == cc_action_t::do_recv) {
        // get a write lock around the recv memory and do the copy
        auto [which, send_ptr] = cc.recv_do.front();
        cc.recv_do.pop();

        lk.unlock();

        sendrecv_t const& move = std::get<sendrecv_t>(graph[which]);

        interval_t recv_interval = move.dst_mem.interval();
        char* recv_ptr = memory.data() + move.dst_mem.offset;

        {
          // 1. Grab a read lock around send_ptr
          auto memlock = memory_lock.acquire({}, {recv_interval});
          // 2. Copy into recv_ptr which should already have a write lock
          std::copy(send_ptr, send_ptr + move.src_mem.size, recv_ptr);
        }

        // 3. Notify complete
        auto& other_manager = get_cpu_device_at(move.src);
        other_manager.cc_from_recv_to_send_notify_complete(which);

        {
          std::unique_lock lk_print(m_print);
          std::cout << "cmd " << which <<
                       " @ cc comm runner " << runner_id << ": " <<
                       graph.get_command(which) << std::endl;
        }

        this->completed_command(which);
      } else if(action == cc_action_t::complete_send) {
        // release the read lock around the send memory
        ident_t which = cc.send_complete.front();
        cc.send_complete.pop();

        lk.unlock();

        sendrecv_t const& move = std::get<sendrecv_t>(graph[which]);

        interval_t send_interval = move.src_mem.interval();
        memory_lock.unlock({send_interval}, {});

        this->completed_command(which);
      } else {
        throw std::runtime_error("should not reach");
      }
    }
  }

  void run() {
    vector<std::thread> runners;
    runners.reserve(num_apply_runners + num_cpucpu_runners);
    for(int i = 0; i != num_apply_runners; ++i) {
      runners.emplace_back([this, i](){ return this->apply_runner(i); });
    }
    for(int i = 0; i != num_cpucpu_runners; ++i) {
      runners.emplace_back([this, i](){ return this->communicate_cpucpu_runner(i); });
    }
    for(std::thread& t: runners) {
      t.join();
    }
  }

private:
  int num_apply_runners;
  int num_cpucpu_runners;

  // ident to number of dependencies remaining until the command can
  // be executed
  vector<int> counts;
  loc_t this_loc;

  // The total number of commands left to execute
  int num_remaining;

  // The memory of this device and a locking device
  // for that memory
  vector<char> memory;
  interval_lock_t memory_lock;

  // Concurrency management
  std::mutex m, m_print;
  std::condition_variable cv;

  // Work for apply runners
  std::queue<int> apply_ready;

  // Work for cpu to cpu communication runners
  //
  // This is how cpu to cpu communication happens:
  // 0. (says sender)   hey, I can do a send           [can_send]
  // 1. (says recver)   hey, I want to do a recv       [can_recv]
  // 2. (says sender)   hey, read from this memory     [start_send]
  //                    (grab read lock)
  // 3. (says recver)   hey, I've finished reading     [do_recv]
  //                    (graph write lock; do copy; release write lock)
  // 4. (notes sender)  (release read lock)            [complete_send]
  enum class cc_action_t { shutdown,
                           can_send,
                           can_recv,
                           start_send,
                           do_recv,
                           complete_send  };
  struct {
    std::queue<int> can_send;
    std::unordered_map<int, int> recv_ready;
    std::queue<tuple<int, char*>> recv_do;
    std::queue<int> send_start;
    std::queue<int> send_complete;
  } cc;
  //cc_t cc;

private:
  cpu_device_t& get_cpu_device_at(loc_t const& loc) {
    return static_cast<cpu_device_t&>(get_device_at(loc));
  }

  void cc_from_send_to_recv_can_send(ident_t const& which) {
    {
      std::unique_lock lk(m);
      cc.recv_ready[which]++;
    }
    cv.notify_all();
  }
  void cc_from_recv_to_send_can_recv(ident_t const& which) {
    {
      std::unique_lock lk(m);
      cc.send_start.push(which);
    }
    cv.notify_all();
  }
  void cc_from_send_to_recv_started_send(ident_t const& which, char* send_ptr) {
    {
      std::unique_lock lk(m);
      cc.recv_do.push({which, send_ptr});
    }
    cv.notify_all();
  }
  void cc_from_recv_to_send_notify_complete(ident_t const& which) {
    {
      std::unique_lock lk(m);
      cc.send_complete.push(which);
    }
    cv.notify_all();
  }

  void completed_command(ident_t const& which) {
    {
      std::unique_lock lk(m);

      num_remaining--;
      auto const& parents = graph.get_parents(which);
      for(ident_t const& parent : parents) {
        command_t const& cmd = graph[parent];
        if(is_signal_location(cmd, this_loc)) {
          counts[parent]--;
          if(counts[parent] == 0) {
            set_ready(cmd, parent);
          }
        }
      }
    }
    cv.notify_all();
  }

  void set_ready(ident_t const& which) {
    return set_ready(graph[which], which);
  }

  void set_ready(command_t const& cmd, ident_t const& which) {
    // invariant: graph[which] == cmd
    if(std::holds_alternative<apply_t>(cmd)) {
      apply_ready.push(which);
    } else if(std::holds_alternative<sendrecv_t>(cmd)) {
      sendrecv_t const& move = std::get<sendrecv_t>(cmd);
      if(move.cc()) {
        if(move.src == this_loc) {
          cc.can_send.push(which);
        } else if(move.dst == this_loc) {
          cc.recv_ready[which]++;
        } else {
          throw std::runtime_error("should not reach ");
        }
      } else {
        throw std::runtime_error("not implemented");
      }
    } else {
      throw std::runtime_error("should not reach");
    }
  }
};


