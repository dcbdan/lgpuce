#pragma once

#include <cstdint>
#include <vector>
#include <functional>
#include <variant>
#include <tuple>

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

