#pragma once

#include <iostream>
#include <vector>
#include <tuple>
#include <algorithm>

using std::vector;
using std::tuple;

struct memorymanager_t {
  memorymanager_t(uint64_t total_size): total_size(total_size) {}

  uint64_t allocate(uint64_t size) {
    if(size > total_size) {
      throw std::runtime_error("not enough memory");
    }
    if(size == 0) {
      throw std::runtime_error("does not support size zero allocate");
    }
    if(allocs.size() == 0) {
      allocs.emplace_back(0, size);
      return 0;
    }
    uint64_t prev = 0;
    auto iter = allocs.begin();
    for(; iter != allocs.end(); ++iter) {
      auto const& [beg, sz] = *iter;
      if(beg-prev >= size) {
        allocs.insert(iter, {prev, size});
        return prev;
      }
      prev = beg + sz;
    }
    if(total_size-prev >= size) {
      allocs.insert(iter, {prev, size});
      return prev;
    }
    throw std::runtime_error("could not find free space");
    return 0;
  }

  void deallocate(uint64_t loc) {
    auto iter = std::find_if(allocs.begin(), allocs.end(), [&loc](auto const& xy) {
      auto const& [beg,_] = xy;
      return beg == loc;
    });
    if(iter == allocs.end()) {
      throw std::runtime_error("could not deallocate");
    }
    allocs.erase(iter);
  }

  void print(std::ostream& out, char newline) const {
    out << "memorymanager_t [0," << total_size << ")" << newline;
    out << "---" << newline;
    for(auto const& [beg,sz]: allocs) {
      out << "[" << beg << "," << beg+sz << ")" << newline;
    }
  }
  void print(std::ostream& out) const {
    return this->print(out, ' ');
  }

private:
  vector<tuple<uint64_t, uint64_t>> allocs;
  uint64_t total_size;
};

std::ostream& operator<<(std::ostream& out, memorymanager_t const& m) {
  m.print(out);
  return out;
}

#ifdef QUICK_TEST
void memormanager_test01() {
  memorymanager_t m(100);
  auto p1 = m.allocate(25);
  std::cout << m << std::endl;
  auto p2 = m.allocate(25);
  std::cout << m << std::endl;
  auto p3 = m.allocate(25);
  std::cout << m << std::endl;
  auto p4 = m.allocate(25);
  std::cout << m << std::endl;
  m.deallocate(p3);
  std::cout << m << std::endl;
  m.deallocate(p2);
  std::cout << m << std::endl;
  m.deallocate(p1);
  std::cout << m << std::endl;
  m.deallocate(p4);
  std::cout << m << std::endl;
}

void memormanager_test02() {
  memorymanager_t m(100);
  auto p1 = m.allocate(25);
  std::cout << m << std::endl;
  auto p2 = m.allocate(25);
  std::cout << m << std::endl;
  auto p3 = m.allocate(25);
  std::cout << m << std::endl;
  auto p4 = m.allocate(25);
  std::cout << m << std::endl;
  m.deallocate(p3);
  std::cout << m << std::endl;
  m.allocate(20);
  std::cout << m << std::endl;
  m.allocate(5);
  std::cout << m << std::endl;
}
#endif
