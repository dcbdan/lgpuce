#pragma once

#include "interval_lock.h"

#include <queue>
#include <unordered_map>

#include <mutex>
#include <condition_variable>

#include <sys/mman.h> // mlock, munlock

struct cluster_t;

// A handler lives on a kernel execution thread.
// * set the device on the thread
// * set up cublas
// * tell cublas to use the thread-specific stream
struct handler_t {
  handler_t(loc_t loc)
    : is_gpu(loc.is_gpu())
  {
    if(is_gpu) {
      cuda_set_device(loc.id);
      if(cublasCreate(&gpu_handle) != CUBLAS_STATUS_SUCCESS){
        throw std::runtime_error("gpu handle creation");
      }
      cublasSetStream(gpu_handle, cudaStreamPerThread);
    }
  }

  ~handler_t() {
    if(is_gpu) {
      cublasDestroy(gpu_handle);
    }
  }

  inline void* operator()() const {
    return (void*)(&gpu_handle);
  }

  inline void synchronize() {
    if(is_gpu) {
      cudaStreamSynchronize(cudaStreamPerThread);
    }
  }

  bool is_gpu;
  cublasHandle_t gpu_handle;
};

struct device_t {
  device_t() = delete;

  ~device_t() {
    if(is_cpu()) {
      cudaFreeHost((void*)memory);
    } else {
      cudaFree((void*)memory);
    }
  }

  device_t(
    cluster_t* manager,
    uint64_t memory_size,
    loc_t this_loc):
      manager(manager),
      memory_size(memory_size),
      this_loc(this_loc)
  {
    if(is_cpu()) {
      // TODO: cudaMallocHost is page-locked, but docs say
      //       this might not be want you want. Should investigate
      //       correct way to manage the big blob of cpu memory
      if(cudaMallocHost((void**)&memory, memory_size) != cudaSuccess) {
        throw std::runtime_error("cudaMallocHost failed");
      }
    } else {
      cuda_set_device(this_loc.id);
      if(cudaMalloc((void**)&memory, memory_size) != cudaSuccess) {
        throw std::runtime_error("cudaMalloc failed");
      }
    }
  }

  void apply_runner(int runner_id) {
    // This is always launched in a new thread, so set the device
    // (this happens inside the handler constructor)
    handler_t handler(this_loc);
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
        apply_t const& apply = std::get<apply_t>((*graph)[which]);
        vector<interval_t> read_intervals, write_intervals;
        vector<void*> read_mems, write_mems;
        for(auto const& mem: apply.read_mems) {
          read_intervals.push_back(mem.interval());
          read_mems.push_back(memory + mem.offset);
        }
        for(auto const& mem: apply.write_mems) {
          write_intervals.push_back(mem.interval());
          write_mems.push_back(memory + mem.offset);
        }
        auto memlock = memory_lock.acquire_and_correct(read_intervals, write_intervals);
        {
          auto e = time_events.log_apply(runner_id);
          apply.op(handler(), read_mems, write_mems);
          handler.synchronize();
        }
      }

      // Tell everyone that a command has been executed
      this->completed_command(which);
    }
  }

  void communicate_runner(int runner_id) {
    // This is always launched in a new thread, so set the device
    // (this happens inside the handler constructor)
    handler_t handler(this_loc);

    comm_action_t action;
    ident_t can_recv_ident;
    while(true) {
      std::unique_lock lk(m);
      cv.wait(lk, [this, &action, &can_recv_ident](){
        if(num_remaining == 0) {
          action = comm_action_t::shutdown;
          return true;
        }
        if(comm.recv_do.size() > 0) {
          action = comm_action_t::do_recv;
          return true;
        }
        if(comm.send_start.size() > 0) {
          action = comm_action_t::start_send;
          return true;
        }
        if(comm.send_complete.size() > 0) {
          action = comm_action_t::complete_send;
          return true;
        }
        if(comm.can_send.size() > 0) {
          action = comm_action_t::can_send;
          return true;
        }
        for(auto const& [ready_recv_ident,cnt]: comm.recv_ready) {
          if(cnt == 2) {
            can_recv_ident = ready_recv_ident;
            action = comm_action_t::can_recv;
            return true;
          }
        }
        return false;
      });

      if(action == comm_action_t::shutdown) {
        return;
      } else if(action == comm_action_t::can_send) {
        // notify the recver that we can do a send
        ident_t which = comm.can_send.front();
        comm.can_send.pop();

        lk.unlock();

        sendrecv_t const& move = std::get<sendrecv_t>((*graph)[which]);
        auto& other_manager = get_device_at(move.dst);
        other_manager.comm_from_send_to_recv_can_send(which);
      } else if(action == comm_action_t::can_recv) {
        // notify the sender that we can do a recv
        auto& which = can_recv_ident;
        comm.recv_ready.erase(which);

        lk.unlock();

        sendrecv_t const& move = std::get<sendrecv_t>((*graph)[which]);
        auto& other_manager = get_device_at(move.src);
        other_manager.comm_from_recv_to_send_can_recv(which);
      } else if(action == comm_action_t::start_send) {
        // as the sender, put a read lock around the memory
        // and give the memory to the recver
        ident_t which = comm.send_start.front();
        comm.send_start.pop();

        lk.unlock();

        sendrecv_t const& move = std::get<sendrecv_t>((*graph)[which]);

        interval_t send_interval = move.src_mem.interval();
        char* send_ptr = memory + move.src_mem.offset;
        memory_lock.lock({send_interval}, {});

        auto& other_manager = get_device_at(move.dst);
        other_manager.comm_from_send_to_recv_started_send(which, send_ptr);
      } else if(action == comm_action_t::do_recv) {
        // get a write lock around the recv memory and do the copy
        auto [which, send_ptr] = comm.recv_do.front();
        comm.recv_do.pop();

        lk.unlock();

        sendrecv_t const& move = std::get<sendrecv_t>((*graph)[which]);

        interval_t recv_interval = move.dst_mem.interval();
        char* recv_ptr = memory + move.dst_mem.offset;

        {
          // TODO: what about cuda streams?

          // 1. Grab a read lock around send_ptr
          auto memlock = memory_lock.acquire({}, {recv_interval});
          // 2. Copy into recv_ptr which should already have a write lock
          {
            auto e = time_events.log_comm(runner_id);
            if(cudaMemcpy(
                (void*)recv_ptr,
                (void*)send_ptr,
                move.src_mem.size,
                cudaMemcpyDefault)
                != cudaSuccess)
            {
              throw std::runtime_error("did not cuda memcpy");
            }
          }
        }

        // 3. Notify complete
        auto& other_manager = get_device_at(move.src);
        other_manager.comm_from_recv_to_send_notify_complete(which);

        this->completed_command(which);
      } else if(action == comm_action_t::complete_send) {
        // release the read lock around the send memory
        ident_t which = comm.send_complete.front();
        comm.send_complete.pop();

        lk.unlock();

        sendrecv_t const& move = std::get<sendrecv_t>((*graph)[which]);

        interval_t send_interval = move.src_mem.interval();
        memory_lock.unlock({send_interval}, {});

        this->completed_command(which);
      } else {
        throw std::runtime_error("should not reach");
      }
    }
  }

  void prepare(graph_t const& g) {
    graph = &g;
    // Reset runner state management
    num_remaining = 0;
    counts = vector<int>(graph->size());
    apply_ready = std::queue<int>();
    comm.reset();

    auto const& info = graph->get_info();
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

  // Assumption: prepare(graph) was called first
  void run(
    int num_apply_runners,
    int num_comm_runners)
  {
    time_events.init(num_apply_runners, num_comm_runners);

    vector<std::thread> runners;
    runners.reserve(num_apply_runners + num_comm_runners);
    for(int i = 0; i != num_apply_runners; ++i) {
      runners.emplace_back([this, i](){ return this->apply_runner(i); });
    }
    for(int i = 0; i != num_comm_runners; ++i) {
      runners.emplace_back([this, i](){ return this->communicate_runner(i); });
    }
    for(std::thread& t: runners) {
      t.join();
    }
  }

  inline bool is_cpu() const {
    return this_loc.device_type == device_type_t::cpu;
  }
  inline bool is_gpu() const {
    return this_loc.device_type == device_type_t::gpu;
  }

#ifdef TIME_EVENTS
  void log_time_events(time_point_t const& base, std::ostream& out) const {
    auto fix = [&base](time_point_t const& t) {
      auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(t - base);
      return duration.count();
    };
    for(auto const& ts: time_events.items) {
      for(auto const& [start,stop,label]: ts) {
        out << fix(start) << "," << fix(stop) << "," << this_loc << "|" << label << std::endl;
      }
    }
  }
#endif

private:
  cluster_t* manager;
  uint64_t memory_size;
  graph_t const* graph;

  loc_t this_loc;

  // ident to number of dependencies remaining until the command can
  // be executed
  vector<int> counts;

  // The total number of commands left to execute
  int num_remaining;

  // The memory of this device and a locking device
  // for that memory
  char* memory;
  interval_lock_t memory_lock;

  // Concurrency management
  std::mutex m;
  std::condition_variable cv;

  // Work for apply runners
  std::queue<int> apply_ready;

  // Work for communication runners
  //
  // This is how communication happens:
  // 0. (says sender)   hey, I can do a send           [can_send]
  // 1. (says recver)   hey, I want to do a recv       [can_recv]
  // 2. (says sender)   hey, read from this memory     [start_send]
  //                    (grab read lock)
  // 3. (says recver)   hey, I've finished reading     [do_recv]
  //                    (graph write lock; do copy; release write lock)
  // 4. (notes sender)  (release read lock)            [complete_send]
  enum class comm_action_t { shutdown,
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

    void reset() {
      can_send = std::queue<int>();
      recv_ready = std::unordered_map<int, int>();
      recv_do = std::queue<tuple<int, char*>>();
      send_start = std::queue<int>();
      send_complete = std::queue<int>();
    }

  } comm;

#ifdef TIME_EVENTS
  struct {
    vector<vector<time_info_t>> items;
    int na;

    void init(int num_app, int num_comm) {
      items = vector<vector<time_info_t>>(num_app + num_comm);
      na = num_app;
    }

    struct raii_t {
      raii_t(vector<vector<time_info_t>>& items, std::string s, int idx):
        items(items),
        s(s),
        idx(idx),
        start(std::chrono::high_resolution_clock::now())
      {}

      ~raii_t() {
        items[idx].push_back(time_info_t {
          .start = start,
          .end   = std::chrono::high_resolution_clock::now(),
          .label = s
        });
      }

      vector<vector<time_info_t>>& items;
      std::string s;
      int idx;
      time_point_t start;
    };

    raii_t log_apply(int idx) {
      return raii_t(items, "apply" + std::to_string(idx), idx);
    }
    raii_t log_comm(int idx) {
      return raii_t(items, "comm" + std::to_string(idx), na + idx);
    }
  } time_events;
#else
  struct {
    void init(int num_app, int num_comm) const {}
    char log_apply(int _) const { return 'a'; };
    char log_comm(int _) const { return 'a'; };
  } time_events;
#endif

private:
  device_t& get_device_at(loc_t const& loc);
  std::mutex& print_lock();

  void comm_from_send_to_recv_can_send(ident_t const& which) {
    {
      std::unique_lock lk(m);
      comm.recv_ready[which]++;
    }
    cv.notify_all();
  }
  void comm_from_recv_to_send_can_recv(ident_t const& which) {
    {
      std::unique_lock lk(m);
      comm.send_start.push(which);
    }
    cv.notify_all();
  }
  void comm_from_send_to_recv_started_send(ident_t const& which, char* send_ptr) {
    {
      std::unique_lock lk(m);
      comm.recv_do.push({which, send_ptr});
    }
    cv.notify_all();
  }
  void comm_from_recv_to_send_notify_complete(ident_t const& which) {
    {
      std::unique_lock lk(m);
      comm.send_complete.push(which);
    }
    cv.notify_all();
  }

  void completed_command(ident_t const& which) {
    {
      std::unique_lock lk(m);

      num_remaining--;
      auto const& parents = graph->get_parents(which);
      for(ident_t const& parent : parents) {
        command_t const& cmd = (*graph)[parent];
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
    return set_ready((*graph)[which], which);
  }

  void set_ready(command_t const& cmd, ident_t const& which) {
    // invariant: graph[which] == cmd
    if(std::holds_alternative<apply_t>(cmd)) {
      apply_ready.push(which);
    } else if(std::holds_alternative<sendrecv_t>(cmd)) {
      sendrecv_t const& move = std::get<sendrecv_t>(cmd);
      if(move.src == this_loc) {
        comm.can_send.push(which);
      } else if(move.dst == this_loc) {
        comm.recv_ready[which]++;
      } else {
        throw std::runtime_error("should not reach ");
      }
    }
  }
};


