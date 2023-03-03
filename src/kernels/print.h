#pragma once

#include "../types.h"
#include "../misc.h"

#include <iostream>

kernel_t gen_print(vector<uint64_t> shape) {
  return [shape](void*, vector<void*> const& inns, vector<void*> const& outs) {
    float* data = (float*)inns[0];
    std::cout << "shape";
    for(auto const& d: shape) {
      std::cout << " " << d;
    }
    std::cout << std::endl;
    uint64_t total = uint64_product(shape);

    if(total < 20) {
      std::cout << "data";
    } else {
      std::cout << "data[0:20]";
    }

    uint64_t nto = 20;
    nto = std::min(nto, total);
    for(uint64_t i = 0; i != nto; ++i) {
      std::cout << " " << data[i];
    }
    std::cout << std::endl;
  };
}

