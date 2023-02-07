#pragma once

#include <cstdint>
#include <vector>
#include <functional>
#include <variant>
#include <tuple>
#include <ostream>

#include "misc.h"

using std::vector;
using std::tuple;

using ident_t = uint64_t;

enum class device_t { cpu, gpu };

struct loc_t {
  device_t device;
  int id;
};

struct mem_t {
  uint64_t offset;
  uint64_t size;
};

using memloc_t = tuple<mem_t, loc_t>;

using kernel_t =
  std::function<
    void(
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
};

using command_t = std::variant<apply_t, sendrecv_t>;

bool operator==(loc_t const& lhs, loc_t const& rhs) {
  return lhs.device == rhs.device && lhs.id == rhs.id;
}

std::ostream& operator<<(std::ostream& out, mem_t mem) {
  //out << "mem_t { .size = " << mem.size << " }";
  out << "mem:" << mem.offset;
  return out;
}

std::ostream& operator<<(std::ostream& out, loc_t loc) {
  //out << "loc_t { .device = <not shown>, .id = " << loc.id << " }";
  out << loc.id;
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


