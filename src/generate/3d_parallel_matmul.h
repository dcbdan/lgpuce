#pragma once

#include "../types.h"
#include "../graph.h"
#include "../kernels.h"
#include "../kernels/print.h"
#include <cstdint>
#include <sys/types.h>
#include <tuple>
#include <utility>
#include <vector>
#include <cmath>

// given a matrix of size m * n, each matrix element is a floating number. The matrix start at address 0
// example: the second element of the first row is at address 0 + sizeof(float)
// 0 <= row_begin < row_end, 0 <= column_begin < column_end, row_end <= n, column_end <= m
// return the vector of tuples, each tuple contains the begin and size of a matrix row in the sliced matrix
// example: matrix_slicing(0, 2, 1, 3) returns {(sizeof(float), sizeof(float) * 2), (sizeof(float) * m + sizeof(float), sizeof(float) * 2)}
std::vector<std::pair<uint64_t, uint64_t>> matrix_slicing(uint64_t m, uint64_t n, uint64_t row_begin, uint64_t row_end, uint64_t column_begin, uint64_t column_end){
  std::vector<std::pair<uint64_t, uint64_t>> ret;
  for (uint64_t i = row_begin; i < row_end; i++){
    uint64_t row_begin_offset = i * m * sizeof(float) + column_begin * sizeof(float);
    uint64_t row_size = (column_end - column_begin) * sizeof(float);
    ret.push_back(std::make_pair(row_begin_offset, row_size));
  }
  return ret;
}
// given the result of matrix_slicing, return the vector of tuples, each tuple contains the begin and size of a matrix row in the further sliced matrix
// parameters: input_matrix is the result of matrix_slicing, column_begin and column_end are the column index of the further sliced matrix
std::vector<std::pair<uint64_t, uint64_t>> matrix_further_slice_by_column(std::vector<std::pair<uint64_t, uint64_t>> input_matrix, uint64_t column_begin, uint64_t column_end){
  std::vector<std::pair<uint64_t, uint64_t>> ret;
  for (auto row : input_matrix){
    uint64_t row_begin_offset = std::get<0>(row) + column_begin * sizeof(float);
    uint64_t row_size = (column_end - column_begin) * sizeof(float);
    ret.push_back(std::make_pair(row_begin_offset, row_size));
  }
  return ret;
}

// given the result of matrix_slicing, return the vector of tuples, each tuple contains the begin and size of a matrix row in the further sliced matrix
// parameters: input_matrix is the result of matrix_slicing, row_begin and row_end are the row index of the further sliced matrix
std::vector<std::pair<uint64_t, uint64_t>> matrix_further_slice_by_row(std::vector<std::pair<uint64_t, uint64_t>> input_matrix, uint64_t row_begin, uint64_t row_end){
  std::vector<std::pair<uint64_t, uint64_t>> ret;
  for (uint64_t i = row_begin; i < row_end; i++){
    ret.push_back(input_matrix[i]);
  }
  return ret;
}


// parameters and relations:
// processor grid dimension: p_1, p_2, p_3 (For GPU 3d matmul this should be the GPU grid)
// Input Matrices Dimension: MxN * N*K = M*K
// m = M/p_1, n = N/p_2, k = K/p_3

// A_il = A(im : im + m - 1, lk : lk + k - 1)          m * k
// B_lj = B(lk : lk + k - 1, jn : jn + n - 1)          k * n
// C_ij = C(im : im + m - 1, jn : jn + n - 1)          m * n

// k_2 = k / p_2
// n_1 = n / p_1
// n_3 = n / p_3
// m_3 = m / p_3

// A_il(j) = A_il( : , jk_2 : jk_2 + k_2 - 1)          m * k_2 = m * (k / p_2)  p_2 blocks of k_2 columns
// B_lj(i) = B_lj( : , in_1 : in_1 + n_1 - 1)          k * n_1 = k * (n / p_1)  p_1 blocks of n_1 columns
// C_ij(l) = C_ij( : , ln_3 : ln_3 + n_3 - 1)          m * n_3 = m * (n / p_3)  p_3 blocks of n_3 columns

// D_ij^l = A_il * B_lj                                m * n
// D_ij^l(l) = D_ij^l( : , ln_3 : ln_3 + n_3 - 1)      m * n_3 = m * (n / p_3)  p_3 blocks of n_3 columns
// we can also slice D_ij^l by row
// D_ij^l(c) = D_ij^l(cm_3 : cm_3 + m_3 - 1, : )       m / p_3 * n = m * (n / p_3)  p_3 blocks of n_3 columns 

graph_t init_mat_GPU(uint64_t p_1, uint64_t p_2, uint64_t p_3, uint64_t m, uint64_t n, uint64_t k, uint64_t num_physical_GPUs)
{
  // change this to the actual VRAM size
  auto gpu_mem_size = 14lu * 1024lu * 1024lu * 1024lu;

  uint64_t M = p_1 * m;
  uint64_t N = p_2 * n;
  uint64_t K = p_3 * k;
  uint64_t lhs_size = sizeof(float)* M * N;
  uint64_t rhs_size = sizeof(float)* N * K;

  uint64_t k_2 = k / p_2;
  uint64_t n_1 = n / p_1;
  uint64_t n_3 = n / p_3;

  uint64_t size_of_A_il = sizeof(float)* m * k;
  uint64_t size_of_B_lj = sizeof(float)* k * n;
  uint64_t size_of_C_ij = sizeof(float)* m * n;

  uint64_t size_of_A_il_j = sizeof(float)* m * k_2;
  uint64_t size_of_B_lj_i = sizeof(float)* k * n_1;
  uint64_t size_of_C_ij_l = sizeof(float)* m * n_3;

  // right now let's just do with 1 cpu but p_1 * p_2 * p_3 GPUs
  loc_t cpu { .device_type = device_type_t::cpu, .id = 0 };
  std::vector<loc_t> gpu_grid;
  for (auto i = 0; i < num_physical_GPUs; ++i){
    gpu_grid.push_back(loc_t{ .device_type = device_type_t::gpu, .id = i});
  }

  mem_t lhs_mem { .offset = 0,        .size = lhs_size };
  mem_t rhs_mem { .offset = lhs_size, .size = rhs_size };

  graph_t g;

  ident_t construct_A = g.insert(
    apply_t{
      .loc = cpu,
      .read_mems = {},
      .write_mems = {lhs_mem},
      .op = gen_constant({M,N}, 1.0) },
    {}
  );

  ident_t construct_B = g.insert(
    apply_t{
      .loc = cpu,
      .read_mems = {},
      .write_mems = {rhs_mem},
      .op = gen_constant({N,K}, 1.0) },
    {}
  );

  // get the starting offset for each virtual GPU memory
  uint64_t num_virtual_GPUs = p_1 * p_2 * p_3;
  uint64_t num_virtual_GPUs_per_physical = std::floor(num_virtual_GPUs / num_physical_GPUs) + 1;
  std::vector<uint64_t> memory_offset_virtual_GPUs;
  uint64_t virtual_GPU_memory_size = std::floor(gpu_mem_size / num_virtual_GPUs_per_physical);
  for (auto i = 0; i < num_virtual_GPUs_per_physical; ++i){
    memory_offset_virtual_GPUs.push_back(virtual_GPU_memory_size * i);
  }

  // for virtual GPU i, j, k in the GPU grid, generate the command to copy the corresponding A_il(j), B_lj(i) to the GPU
  for (auto i = 0; i < p_1; ++i){
    for (auto j = 0; j < p_2; ++j){
      for (auto l = 0; l < p_3; ++l){
        auto virtual_GPU_id = i * p_2 * p_3 + j * p_2 + l;
        auto physical_GPU_id = std::floor(virtual_GPU_id / num_virtual_GPUs_per_physical);
        auto virtual_GPU_id_in_physical = virtual_GPU_id % num_virtual_GPUs_per_physical;
        auto virtual_GPU_memory_offset = memory_offset_virtual_GPUs[virtual_GPU_id_in_physical];

        auto A_il = matrix_slicing(M, N, i*m, i*m + m - 1, l*k, l*k + k - 1);
        auto A_il_j = matrix_further_slice_by_row(A_il, j*k_2, j*k_2 + k_2 - 1);

        auto B_lj = matrix_slicing(N, K, l*k, l*k + k - 1, j*n, j*n + n - 1);
        auto B_lj_i = matrix_further_slice_by_row(B_lj, i*n_1, i*n_1 + n_1 - 1);
        // moving A_il(j) to GPU
        for (auto row : A_il_j){
          g.insert(
            sendrecv_t{
              .src = cpu,
              .dst = gpu_grid[physical_GPU_id],
              .src_mem = {mem_t{.offset = row.first, .size = row.second}},
              .dst_mem = {mem_t{.offset = virtual_GPU_memory_offset + j * size_of_A_il_j + row.first, .size = row.second}}
              },
              {construct_A}
          );
        }

        // moving B_lj(i) to GPU
        for (auto row: B_lj_i){
          g.insert(
            sendrecv_t{
              .src = cpu,
              .dst = gpu_grid[physical_GPU_id],
              .src_mem = {mem_t{.offset = row.first, .size = row.second}},
              .dst_mem = {mem_t{.offset = virtual_GPU_memory_offset + size_of_A_il + i * size_of_B_lj_i + row.first, .size = row.second}}
              },
              {construct_B}
          );
        }
      }
    }
  }
  return g;
}

// generate graph for actually performing the 3d matrix multiplication after the data has been copied to the GPU by the previous graph output by init_mat_GPU
graph_t matmul_3d(uint64_t p_1, uint64_t p_2, uint64_t p_3, uint64_t m, uint64_t n, uint64_t k, uint64_t num_physical_GPUs){
  // change this to the actual VRAM size
  auto gpu_mem_size = 14lu * 1024lu * 1024lu * 1024lu;

  uint64_t M = p_1 * m;
  uint64_t N = p_2 * n;
  uint64_t K = p_3 * k;
  uint64_t lhs_size = sizeof(float)* M * N;
  uint64_t rhs_size = sizeof(float)* N * K;

  uint64_t k_2 = k / p_2;
  uint64_t n_1 = n / p_1;
  uint64_t n_3 = n / p_3;

  uint64_t size_of_A_il = sizeof(float)* m * k;
  uint64_t size_of_B_lj = sizeof(float)* k * n;
  uint64_t size_of_C_ij = sizeof(float)* m * n;

  uint64_t size_of_A_il_j = sizeof(float)* m * k_2;
  uint64_t size_of_B_lj_i = sizeof(float)* k * n_1;
  uint64_t size_of_C_ij_l = sizeof(float)* m * n_3;
  // D_ij^l = A_il * B_lj
  uint64_t size_of_D_ij_l = sizeof(float)* m * n;

  // right now let's just do with 1 cpu but p_1 * p_2 * p_3 GPUs
  loc_t cpu { .device_type = device_type_t::cpu, .id = 0 };
  std::vector<loc_t> gpu_grid;
  for (auto i = 0; i < num_physical_GPUs; ++i){
    gpu_grid.push_back(loc_t{ .device_type = device_type_t::gpu, .id = i});
  }

  // get the starting offset for each virtual GPU memory
  uint64_t num_virtual_GPUs = p_1 * p_2 * p_3;
  uint64_t num_virtual_GPUs_per_physical = std::floor(num_virtual_GPUs / num_physical_GPUs) + 1;
  std::vector<uint64_t> memory_offset_virtual_GPUs;
  uint64_t virtual_GPU_memory_size = std::floor(gpu_mem_size / num_virtual_GPUs_per_physical);
  for (auto i = 0; i < num_virtual_GPUs_per_physical; ++i){
    memory_offset_virtual_GPUs.push_back(virtual_GPU_memory_size * i);
  }

  graph_t g;

  // create a vector of size p_1 * p_3 to store all the communications of A_il_j
  // each element in the vector is a vector of size p_2
  std::vector<std::vector<ident_t>> A_il_j_comms;
  for (auto i = 0; i < p_1 * p_3; ++i){
    A_il_j_comms.push_back(std::vector<ident_t>());
  }

  // create a vector of size p_2 * p_3 to store all the communications of B_lj_i
  // each element in the vector is a vector of size p_1
  std::vector<std::vector<ident_t>> B_lj_i_comms;
  for (auto i = 0; i < p_2 * p_3; ++i){
    B_lj_i_comms.push_back(std::vector<ident_t>());
  }

  // create a vector of size p_1 * p_2 to store all the communications of C_ij_l
  // each element in the vector is a vector of size p_3
  std::vector<std::vector<ident_t>> C_ij_l_comms;
  for (auto i = 0; i < p_1 * p_2; ++i){
    C_ij_l_comms.push_back(std::vector<ident_t>());
  }

  // for processor (i, j ,l), it has A_il_j and we need to get A_il from all the A_il_j from other GPUs with the same i, l but different j
  // for processor (i, j ,l), it has B_lj_i and we need to get B_lj from all the B_lj_i from other GPUs with the same l, j but different i
  for (auto i = 0; i < p_1; ++i){
    for (auto j = 0; j < p_2; ++j){
      for (auto l = 0; l < p_3; ++l){
        auto virtual_GPU_id = i * p_2 * p_3 + j * p_2 + l;
        auto physical_GPU_id = std::floor(virtual_GPU_id / num_virtual_GPUs_per_physical);
        auto virtual_GPU_id_in_physical = virtual_GPU_id % num_virtual_GPUs_per_physical;
        auto virtual_GPU_memory_offset = memory_offset_virtual_GPUs[virtual_GPU_id_in_physical];

        // keep track of the offset for each tensor
        auto my_offset_start = virtual_GPU_memory_offset;
        auto my_offset_A_il_j = virtual_GPU_memory_offset + j * size_of_A_il_j;
        auto my_offset_B_lj_i = virtual_GPU_memory_offset + size_of_A_il + i * size_of_B_lj_i;
        auto my_offset_D_ij_l = virtual_GPU_memory_offset + size_of_A_il + size_of_B_lj;
        auto my_offset_D_ij_l_r = my_offset_D_ij_l + size_of_D_ij_l + l * size_of_C_ij_l;
        auto my_offset_C_ij_l = my_offset_D_ij_l + size_of_D_ij_l * 2;

        // every processor needs to send its A_il_j to the processor p_2 - 1 times (no need to send to itself) 
        // every processor needs to send its B_lj_i to the processor p_1 - 1 times (no need to send to itself)
        
        // get A_il from all the A_il_j from other GPUs with the same i, l but different j
        for (auto j_2 = 0; j_2 < p_2; ++j_2){
          if (j_2 == j){
            continue;
          }
          
          auto other_virtual_GPU_id = i * p_2 * p_3 + j_2 * p_2 + l;
          auto other_physical_GPU_id = std::floor(other_virtual_GPU_id / num_virtual_GPUs_per_physical);
          // the starting offset for the other virtual GPU is different but the relative offset is the same
          auto other_virtual_GPU_memory_offset = memory_offset_virtual_GPUs[other_virtual_GPU_id % num_virtual_GPUs_per_physical] 
                                                                  + j * size_of_A_il_j;
          ident_t my_move_A = g.insert(
            sendrecv_t{
              .src = gpu_grid[physical_GPU_id],
              .dst = gpu_grid[other_physical_GPU_id],
              .src_mem = {mem_t{.offset = my_offset_A_il_j, .size = size_of_A_il_j}},
              .dst_mem = {mem_t{.offset = other_virtual_GPU_memory_offset, .size = size_of_A_il_j}}
              }
          );

          // add the communication to the vector
          A_il_j_comms[i * p_1 + l * p_3].push_back(my_move_A);
        }

        // get B_lj from all the B_lj_i from other GPUs with the same l, j but different i
        for (auto i_2 = 0; i_2 < p_1; ++i_2){
          if (i_2 == i){
            continue;
          }
          auto other_virtual_GPU_id = i_2 * p_2 * p_3 + j * p_2 + l;
          auto other_physical_GPU_id = std::floor(other_virtual_GPU_id / num_virtual_GPUs_per_physical);
          auto other_virtual_GPU_memory_offset = memory_offset_virtual_GPUs[other_virtual_GPU_id % num_virtual_GPUs_per_physical] 
                                                                  + size_of_A_il + i * size_of_B_lj_i;
          ident_t my_move_B = g.insert(
            sendrecv_t{
              .src = gpu_grid[physical_GPU_id],
              .dst = gpu_grid[other_physical_GPU_id],
              .src_mem = {mem_t{.offset = my_offset_B_lj_i, .size = size_of_B_lj_i}},
              .dst_mem = {mem_t{.offset = other_virtual_GPU_memory_offset, .size = size_of_B_lj_i}}
              }
          );

          // add the communication to the vector
          B_lj_i_comms[j * p_2 + l * p_3].push_back(my_move_B);
        }

        // after the communications are done, perform the computation A_il * B_lj = D_ij^l
        // my_requirement is a vector with all elements from A_il_j_comms[i * p_1 + l * p_3] and B_lj_i_comms[j * p_2 + l * p_3]
        std::vector<ident_t> my_requirement;
        for (auto A_comm: A_il_j_comms[i * p_1 + l * p_3]){
          my_requirement.push_back(A_comm);
        }
        for (auto B_comm: B_lj_i_comms[j * p_2 + l * p_3]){
          my_requirement.push_back(B_comm);
        }

        ident_t my_compute = g.insert(
          apply_t {
            .loc = gpu_grid[physical_GPU_id],
            .read_mems = {mem_t{.offset = my_offset_start, .size = size_of_A_il}, {mem_t {.offset = my_offset_start + size_of_A_il, .size = size_of_B_lj}}},
            .write_mems = {mem_t{.offset = my_offset_start + size_of_A_il + size_of_B_lj, .size = size_of_D_ij_l}},
            .op = gen_gpu_matmul(m,k,n)},
          my_requirement
        );

        // after the computation is done, we need to send the result D_ij^l to the processor (i, j, l + 1)
        // get all D_ij^r(l) from other GPUs with the same i, j but different l
        for (auto l_2 = 0; l_2 < p_3; ++l_2){
          auto other_virtual_GPU_id = i * p_2 * p_3 + j * p_2 + l_2;
          auto other_physical_GPU_id = std::floor(other_virtual_GPU_id / num_virtual_GPUs_per_physical);
          auto other_virtual_GPU_memory_offset = memory_offset_virtual_GPUs[other_virtual_GPU_id % num_virtual_GPUs_per_physical] 
                                                                  + size_of_A_il + size_of_B_lj + size_of_D_ij_l + l * size_of_D_ij_l;
          ident_t my_move_D = g.insert(
            sendrecv_t{
              .src = gpu_grid[physical_GPU_id],
              .dst = gpu_grid[other_physical_GPU_id],
              .src_mem = {mem_t{.offset = my_offset_C_ij_l, .size = size_of_D_ij_l}},
              .dst_mem = {mem_t{.offset = other_virtual_GPU_memory_offset, .size = size_of_D_ij_l}}
              }
          );

          // add the communication to the vector
          C_ij_l_comms[i * p_1 + j * p_2].push_back(my_move_D);
        }


        // Finally aggregate all the communications
        // do multiple matrix adds
        // my_requirement_C is a vector with all elements from C_ij_l_comms[i * p_1 + j * p_2]
        std::vector<ident_t> my_requirement_C = C_ij_l_comms[i * p_1 + j * p_2];
        for (auto l_2 = 0; l_2 < p_3; ++l_2){
          
          ident_t my_compute = g.insert(
            apply_t {
              .loc = gpu_grid[physical_GPU_id],
              .read_mems = {mem_t{.offset = my_offset_start, .size = size_of_A_il}, {mem_t {.offset = my_offset_start + size_of_A_il, .size = size_of_B_lj}}},
              .write_mems = {mem_t{.offset = my_offset_start + size_of_A_il + size_of_B_lj, .size = size_of_D_ij_l}},
              .op = gen_gpu_matmul(m,k,n)},
            my_requirement
          );
        }

      }
    }
  }
  return g;
}