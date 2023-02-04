#pragma once

#include "types.h"

uint64_t uint64_product(vector<uint64_t> const& xs) {
  uint64_t total = 1;
  for(auto const& x: xs) {
    total *= x;
  }
  return total;
}
