#pragma once

#include "../types.h"
#include "../misc.h"

#include <iostream>

kernel_t gen_print(vector<uint64_t> shape) {
  return [shape](vector<void*> const& inns, vector<void*> const& outs) {
    float* data = (float*)inns[0];
    std::cout << "shape";
    for(auto const& d: shape) {
      std::cout << " " << d;
    }
    std::cout << std::endl;
    uint64_t total = uint64_product(shape);

    std::cout << "data";
    for(int i = 0; i != total; ++i) {
      std::cout << " " << data[i];
    }
    std::cout << std::endl;
  };
}

