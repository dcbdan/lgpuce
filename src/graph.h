#include "types.h"
#include "misc.h"

class graph_t {
public:
  command_t const& get_command(ident_t id) const {
    return info[id].cmd;
  }
  vector<ident_t> const& get_parents(ident_t id) const {
    return info[id].parents;
  }

  // A command can be executed only after children commands
  // have been completed
  vector<ident_t> const& get_children(ident_t id) const {
    return info[id].children;
  }

  ident_t insert(command_t command, vector<ident_t> const& children) {
    info.push_back(
      info_t{ .cmd = command, .parents = {}, .children = children});
    return info.size() - 1;
  }

  ident_t insert(command_t command) {
    return this->insert(command, {});
  }

  void print(std::ostream& out) {
    out << "--- graph_t print ---"  << std::endl;
    for(uint64_t i = 0; i != info.size(); ++i) {
      auto const& [cmd, _, children] = info[i];
      out << i << ": " << cmd << std::endl;
      out << "  depends on ";
      print_vec(out, children);
      out << std::endl;
    }
  }

private:
  struct info_t {
    command_t cmd;
    vector<ident_t> parents;
    vector<ident_t> children;
  };
  vector<info_t> info;
};
