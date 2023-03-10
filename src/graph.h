#pragma once

#include "types.h"
#include "misc.h"

// A note about graph dependencies:
//
// Let every command have a set of signal locations
//   apply:    loc
//   sendrecv: src,dst
// A command is done when it is done at all signal locations.
//
// A dependency is only valid if the child's signal locations
// intercept with the parent's signal locations.
//
// This allows things like
//   cmd1 = (send x1 from a to b)
//   cmd2 = (send x2 from a to b)
//   cmd3 = (send x3 from a to b)
//   have cmd1 then cmd2 then cmd3 happen.
// An alternative dependency constraint would be to have
//   (child)  (parent)
//   apply -> apply    (must live on the same loc)
//   apply -> send     (src must be apply loc)
//   send  -> apply    (dsstloc must be apply loc)
//   send  -> send     (from dst must be to src)
// In this case,
//   * cmd2 could not depend on cmd1 because cmd1.dst != cmd2.src
//   * cmd1,cmd2,cmd3 could not have a dependency structure on node a or b
//
// The benefit of restricting dependencies across signal locations is that
// the device only has to know what things it executed
bool is_signal_location(command_t const& cmd, loc_t const& loc) {
  if(std::holds_alternative<apply_t>(cmd)) {
    apply_t const& apply = std::get<apply_t>(cmd);
    return loc == apply.loc;
  } else if(std::holds_alternative<sendrecv_t>(cmd)) {
    sendrecv_t const& move = std::get<sendrecv_t>(cmd);
    return move.src == loc || move.dst == loc;
  }
  throw std::runtime_error("should not reach");
  return false;
}

vector<loc_t> signal_locations(command_t const& cmd) {
  if(std::holds_alternative<apply_t>(cmd)) {
    apply_t const& apply = std::get<apply_t>(cmd);
    return {apply.loc};
  } else if(std::holds_alternative<sendrecv_t>(cmd)) {
    sendrecv_t const& move = std::get<sendrecv_t>(cmd);
    return {move.src, move.dst};
  }
  throw std::runtime_error("should not reach");
  return {};
}

struct graph_t {
  struct info_t {
    command_t cmd;
    unordered_set<ident_t> parents;
    unordered_set<ident_t> children;
  };

  command_t const& get_command(ident_t const& id) const {
    return info[id].cmd;
  }

  command_t const& operator[](ident_t const& id) const {
    return get_command(id);
  }

  unordered_set<ident_t> const& get_parents(ident_t id) const {
    return info[id].parents;
  }

  // A command can be executed only after children commands
  // have been completed
  unordered_set<ident_t> const& get_children(ident_t id) const {
    return info[id].children;
  }

  ident_t insert(command_t command, vector<ident_t> const& children) {
    if(!valid_children(command, children)) {
      throw std::runtime_error("all children are location specific");
    }
    info.push_back(
      info_t{
        .cmd = command,
        .parents = {},
        .children = unordered_set<ident_t>(children.begin(), children.end())
      });

    ident_t ret = info.size() - 1;
    for(auto const& child: children) {
      info[child].parents.insert(ret);
    }
    return ret;
  }

  ident_t insert(command_t command) {
    return this->insert(command, {});
  }

  void print(std::ostream& out) const {
    out << "--- graph_t print ---"  << std::endl;
    for(uint64_t i = 0; i != info.size(); ++i) {
      auto const& [cmd, _, children] = info[i];
      out << i << ": " << cmd << std::endl;
      out << "  depends on ";
      print_set(out, children);
      out << std::endl;
    }
  }

  vector<int> loc_dependencies(loc_t loc) const {
    vector<int> ret(info.size(), 0);
    for(auto const& [cmd, _, children]: info) {
      if(std::holds_alternative<apply_t>(cmd)) {
        apply_t const& apply = std::get<apply_t>(cmd);
        //
      } else if(std::holds_alternative<sendrecv_t>(cmd)) {
        sendrecv_t const& move = std::get<sendrecv_t>(cmd);
        //
      } else {
        throw std::runtime_error("should not reach");
      }
    }
  }

  vector<info_t> const& get_info() const { return info; }
  uint64_t size() const { return info.size(); }

  uint64_t memory_size(loc_t const& loc) const {
    auto get_memories = [&loc](command_t const& cmd) {
      vector<mem_t> ret;
      if(std::holds_alternative<apply_t>(cmd)) {
        apply_t const& apply = std::get<apply_t>(cmd);
        if (apply.loc == loc) {
          ret.insert(ret.end(), apply.read_mems.begin(),  apply.read_mems.end());
          ret.insert(ret.end(), apply.write_mems.begin(), apply.write_mems.end());
        }
        return ret;
      } else if(std::holds_alternative<sendrecv_t>(cmd)) {
        sendrecv_t const& move = std::get<sendrecv_t>(cmd);
        if(move.src == loc) {
          ret.push_back(move.src_mem);
        }
        if(move.dst == loc){
          ret.push_back(move.dst_mem);
        } // It shouldn't be allowed for move.dst == move.src  ...
        return ret;
      } else {
        throw std::runtime_error("should not reach");
        return ret;
      }
    };
    uint64_t total = 0;
    for(auto const& [cmd, _0, _1]: info) {
      for(mem_t const& mem: get_memories(cmd)) {
        total = std::max(total, mem.offset + mem.size);
      }
    }
    return total;
  }

  int num_cpus() const {
    return num_dev_type(device_type_t::cpu);
  }
  int num_gpus() const {
    return num_dev_type(device_type_t::gpu);
  }

private:
  vector<info_t> info;

  bool valid_children(command_t command, vector<ident_t> const& children) const {
    for(ident_t const& id: children) {
      auto const& [child, _0, _1] = info[id];
      if(!valid_dependency(child, command)) {
        return false;
      }
    }
    return true;
  }

  bool valid_dependency(command_t child, command_t parent) const {
    auto xs = signal_locations(child);
    auto ys = signal_locations(parent);
    for(auto const& x: xs) {
      for(auto const& y: ys) {
        if(x == y) {
          return true;
        }
      }
    }
    return false;
  }

  int num_dev_type(device_type_t const& dd) const {
    int ret = -1;
    auto update = [&](loc_t const& loc) {
      if(loc.device_type == dd) {
        ret = std::max(loc.id, ret);
      }
    };
    for(auto const& [cmd, _0, _1]: this->info) {
      if(std::holds_alternative<apply_t>(cmd)) {
        apply_t const& apply = std::get<apply_t>(cmd);
        update(apply.loc);
      } else if(std::holds_alternative<sendrecv_t>(cmd)) {
        sendrecv_t const& move = std::get<sendrecv_t>(cmd);
        update(move.src);
        update(move.dst);
      }
    }

    return ret + 1;
  }
};
