#pragma once

#include <vector>

#include <cuda_runtime.h>

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

template <typename T>
void print_set(std::ostream& out, std::unordered_set<T> const& xs) {
  if(xs.size() == 0) {
    out << "[]";
  } else if(xs.size() == 1) {
    out << "[" << *xs.begin() << "]";
  } else {
    auto iter = xs.begin();
    out << "[" << *iter;
    iter++;
    for(; iter != xs.end(); ++iter) {
      out << "," << *iter;
    }
    out << "]";
  }
}

int cuda_get_device_count() {
  int ret;
  if(cudaGetDeviceCount(&ret) != cudaSuccess) {
    throw std::runtime_error("cuda get device count");
  }
  return ret;
}

void cuda_set_device(int which) {
  if(cudaSetDevice(which) != cudaSuccess) {
    throw std::runtime_error("cuda set device");
  }
}

int cuda_get_device() {
  int ret;
  if(cudaGetDevice(&ret) != cudaSuccess) {
    throw std::runtime_error("cuda get device");
  }
  return ret;
}

