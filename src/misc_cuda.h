#pragma once

#include <cuda_runtime.h>

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

bool cuda_device_can_access_peer(int d0, int d1) {
  int ret;
  if(cudaDeviceCanAccessPeer(&ret, d0, d1)) {
    throw std::runtime_error("cuda device can access peer");
  }
  return bool(ret);
}

void cuda_device_enable_peer_access(int src, int dst) {
  int before = cuda_get_device();
  cuda_set_device(src);
  if(cudaDeviceEnablePeerAccess(dst, 0) != cudaSuccess) {
    throw std::runtime_error("could not enable peer access");
  }
  cuda_set_device(before);
}

