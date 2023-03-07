#pragma once

#include <cstdint>
#include <vector>
#include <unordered_set>
#include <functional>
#include <variant>
#include <tuple>
#include <ostream>
#include <chrono>

#include "misc.h"

#include <cuda_runtime.h>
#include "cublas_v2.h"

#include <iostream> // TODO
#define DOUT(x) std::cout << x << std::endl;
#define DLINEOUT(x) std::cout << __LINE__ << " " << x << std::endl;
#define DLINE DLINEOUT(' ')
#define DLINEFILEOUT(x) std::cout << __FILE__ << " @ " << __LINE__ << ": " << x << std::endl;
#define DLINEFILE DLINEFILEOUT(' ')

using std::vector;
using std::unordered_set;
using std::tuple;

using ident_t = uint64_t;

enum class device_type_t { cpu, gpu };

struct loc_t {
  device_type_t device_type;
  int id;

};

loc_t make_cpu_loc(int id) {
  return loc_t { .device_type = device_type_t::cpu, id = id };
}
loc_t make_gpu_loc(int id) {
  return loc_t { .device_type = device_type_t::gpu, id = id };
}

using interval_t = tuple<uint64_t, uint64_t>;

struct mem_t {
  uint64_t offset;
  uint64_t size;

  interval_t interval() const {
    return {offset, offset + size};
  }
};

using memloc_t = tuple<mem_t, loc_t>;

using kernel_t =
  std::function<
    void(
      void*,
      vector<void*> const&,
      vector<void*> const&)>;

struct apply_t {
  loc_t loc;
  vector<mem_t> read_mems;
  vector<mem_t> write_mems;
  kernel_t op;
};

struct sendrecv_t {
  loc_t src;
  loc_t dst;
  mem_t src_mem;
  mem_t dst_mem;

  // c = cpu
  // g = gpu
  // cc = cpu to cpu
  // cg = cpu to gpu
  // gc = gpu to cpu
  // gg = gpu to gpu
  bool cc() const {
    return src.device_type == device_type_t::cpu &&
           dst.device_type == device_type_t::cpu;
  }
  bool cg() const {
    return src.device_type == device_type_t::cpu &&
           dst.device_type == device_type_t::gpu;
  }
  bool gc() const {
    return src.device_type == device_type_t::gpu &&
           dst.device_type == device_type_t::cpu;
  }
  bool gg() const {
    return src.device_type == device_type_t::gpu &&
           dst.device_type == device_type_t::gpu;
  }

  cudaMemcpyKind kind() const {
    if(cc()) {
      return cudaMemcpyHostToHost;
    } else if(cg()) {
      return cudaMemcpyHostToDevice;
    } else if(gc()) {
      return cudaMemcpyDeviceToHost;
    } else if(gg()) {
      return cudaMemcpyDeviceToDevice;
    } else {
      throw std::runtime_error("should not reach");
    }
  }
};

using command_t = std::variant<apply_t, sendrecv_t>;

bool operator==(loc_t const& lhs, loc_t const& rhs) {
  return lhs.device_type == rhs.device_type && lhs.id == rhs.id;
}

std::ostream& operator<<(std::ostream& out, mem_t mem) {
  out << "mem[" << mem.offset << "," << mem.offset + mem.size << ")";
  return out;
}

std::ostream& operator<<(std::ostream& out, loc_t loc) {
  if(loc.device_type == device_type_t::cpu) {
    out << "cpu[" << loc.id << "]";
  } else {
    out << "gpu[" << loc.id << "]";
  }
  return out;
}

std::ostream& operator<<(std::ostream& out, apply_t const& a) {
  out << "apply_t { .loc = " << a.loc << ", .read_mems = ";
  print_vec(out, a.read_mems);
  out << ", .write_mems = ";
  print_vec(out, a.write_mems);
  out << " }";
  return out;
}

std::ostream& operator<<(std::ostream& out, sendrecv_t const& a) {
  out << "sendrecv_t { .src = " << a.src << ", .dst = " << a.dst << ", ";
  out << ".src_mem = " << a.src_mem << ", .dst_mem = " << a.dst_mem << " }";
  return out;
}

// Make sure to place this function after the command types
// otherwise the command type will be converted to a command_t and
// an infinite loop will happen
std::ostream& operator<<(std::ostream& out, command_t const& cmd) {
  if(std::holds_alternative<apply_t>(cmd)) {
    out << std::get<apply_t>(cmd);
  } else if(std::holds_alternative<sendrecv_t>(cmd)) {
    out << std::get<sendrecv_t>(cmd);
  } else {
    throw std::runtime_error("should not reach");
  }
  return out;
}

using time_point_t = decltype(std::chrono::high_resolution_clock::now());

#define TIME_EVENTS

#ifdef TIME_EVENTS
#include <string>


struct time_info_t {
  time_point_t start;
  time_point_t end;
  std::string label;
};

//std::ostream& operator<<(std::ostream& out, time_info_t const& x) {
//  auto start = std::chrono::duration_cast<std::chrono::nanoseconds>(x.start - 0);
//  auto end   = std::chrono::duration_cast<std::chrono::nanoseconds>(x.end - 0);
//  std::cout << start.count() << "," << end.count() << "," << x.label;
//  return out;
//}

#endif
