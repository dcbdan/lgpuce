#pragma once

#include "../types.h"

namespace sim {

struct connection_t {
  uint64_t bandwidth; // bytes per second
  loc_t src;
  loc_t dst;
};

struct device_t {
  loc_t loc;
  uint64_t compute; // flop per second per capacity
  int capacity;     // the number of workers or streams
};

struct cluster_t {
  void insert_device(device_t const& d) {
    if(to_device.count(d.loc) > 0) {
      throw std::runtime_error("already contain this loc");
    }

    devices.push_back(d);
    to_device.insert({d.loc, devices.size()-1});
  }

  void insert_connection(connection_t const& c) {
    if(!has_device(c.src) || !has_device(c.dst)) {
      throw std::runtime_error("must insert device first");
    }

    auto iter = to_connection.find({c.src, c.dst});
    if(iter != to_connection.end()) {
      // There is only one unidirectional connection
      // between devices, and adding more connections just
      // increases the bandwidth.
      // (This is what was observed with cudaMemcpy over
      //  nvlink on the anton-j0 node.)
      connection_t& prev_c = connections[iter->second];
      prev_c.bandwidth += c.bandwidth;
    } else {
      connections.push_back(c);
      to_connection.insert({{c.src, c.dst}, connections.size()-1});
    }
  }

  bool has_device(loc_t const& l) const {
    return to_device.count(l) > 0;
  }

  bool has_connection(loc_t const& src, loc_t const& dst) const {
    return to_connection.count({src, dst}) > 0;
  }

  float cost_command(command_t const& cmd) const {
    if(std::holds_alternative<apply_t>(cmd)) {
      apply_t const& apply = std::get<apply_t>(cmd);
      uint64_t compute = device_compute(apply.loc);
      uint64_t flops   = apply.op.flops;
      return (1.0 / compute) * flops;
    } else if(std::holds_alternative<sendrecv_t>(cmd)) {
      sendrecv_t const& move = std::get<sendrecv_t>(cmd);
      uint64_t bytes     = move.src_mem.size;
      uint64_t bandwidth = connection_bandwidth(move.src, move.dst);
      return (1.0 / bandwidth) * bytes;
    } else {
    throw std::runtime_error("should not reach");
    return -1.0;
    }
  }

  int num_devices() const {
    return devices.size();
  }

  int num_connections() const {
    return connections.size();
  }

  int device_capacity(loc_t const& l) const {
    return devices[to_device.at(l)].capacity;
  }

  uint64_t connection_bandwidth(loc_t const& src, loc_t const& dst) const {
    return connections[to_connection.at({src, dst})].bandwidth;
  }

  uint64_t device_compute(loc_t const& l) const {
    return devices[to_device.at(l)].compute;
  }

private:
  vector<device_t> devices;
  vector<connection_t> connections;

  unordered_map<loc_t, int> to_device;
  unordered_map<tuple<loc_t, loc_t>, int> to_connection;
};

}
