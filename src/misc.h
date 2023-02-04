#pragma once

#include <vector>

uint64_t uint64_product(std::vector<uint64_t> const& xs) {
  uint64_t total = 1;
  for(auto const& x: xs) {
    total *= x;
  }
  return total;
}

template <typename T>
void print_vec(std::ostream& out, std::vector<T> const& xs) {
  if(xs.size() == 0) {
    out << "[]";
  } else if(xs.size() == 1) {
    out << "[" << xs[0] << "]";
  } else {
    out << "[" << xs[0];
    for(auto iter = xs.begin() + 1; iter != xs.end(); ++iter) {
      out << "," << *iter;
    }
    out << "]";
  }
}
