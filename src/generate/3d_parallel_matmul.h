#pragma once

#include "../types.h"
#include "../graph.h"
#include "../kernels.h"
#include "../kernels/print.h"
#include <cstdint>
#include <iostream>
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
  for (uint64_t i = row_begin; i <= row_end; i++){
    uint64_t row_begin_offset = i * m * sizeof(float) + column_begin * sizeof(float);
    uint64_t row_size = (column_end - column_begin + 1) * sizeof(float);
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
    uint64_t row_size = (column_end - column_begin + 1) * sizeof(float);
    ret.push_back(std::make_pair(row_begin_offset, row_size));
  }
  return ret;
}

// given the result of matrix_slicing, return the vector of tuples, each tuple contains the begin and size of a matrix row in the further sliced matrix
// parameters: input_matrix is the result of matrix_slicing, row_begin and row_end are the row index of the further sliced matrix
std::vector<std::pair<uint64_t, uint64_t>> matrix_further_slice_by_row(std::vector<std::pair<uint64_t, uint64_t>> input_matrix, uint64_t row_begin, uint64_t row_end){
  std::vector<std::pair<uint64_t, uint64_t>> ret;
  for (uint64_t i = row_begin; i <= row_end; i++){
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
  uint64_t size_of_D_ij_l = sizeof(float)* m * n;

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
  uint64_t num_virtual_GPUs_per_physical = (num_virtual_GPUs % num_physical_GPUs == 0) ? std::floor(num_virtual_GPUs / num_physical_GPUs) 
                                              : std::floor(num_virtual_GPUs / num_physical_GPUs) + 1;
  std::vector<uint64_t> memory_offset_virtual_GPUs;
  uint64_t virtual_GPU_memory_size = std::floor(gpu_mem_size / num_virtual_GPUs_per_physical);
  // Check if there is enough memory on the GPU to perform 3d parallel
  if (virtual_GPU_memory_size < size_of_A_il_j + size_of_B_lj_i + + size_of_D_ij_l * 2 + size_of_C_ij_l){
    throw std::runtime_error("*** Not enough memory on GPU to perfrom 3d parallel ***\n");
    exit(1);
  }

  for (auto i = 0; i < num_virtual_GPUs_per_physical; ++i){
    memory_offset_virtual_GPUs.push_back(virtual_GPU_memory_size * i);
  }

  // for virtual GPU i, j, k in the GPU grid, generate the command to copy the corresponding A_il(j), B_lj(i) to the GPU
  for (auto i = 0; i < p_1; ++i){
    for (auto j = 0; j < p_2; ++j){
      for (auto l = 0; l < p_3; ++l){
        auto virtual_GPU_id = i * p_2 * p_3 + j * p_3 + l;
        // print virtual_GPU_id
        // std::cout << virtual_GPU_id << std::endl;

        auto physical_GPU_id = std::floor(virtual_GPU_id / num_virtual_GPUs_per_physical);
        auto virtual_GPU_id_in_physical = virtual_GPU_id % num_virtual_GPUs_per_physical;
        auto virtual_GPU_memory_offset = memory_offset_virtual_GPUs[virtual_GPU_id_in_physical];

        auto A_il = matrix_slicing(M, N, i*m, i*m + m - 1, l*k, l*k + k - 1);
        auto A_il_j = matrix_further_slice_by_row(A_il, j*k_2, j*k_2 + k_2 - 1);

        // print size of A_il
        // std::cout << A_il.size() << std::endl;
        // print first element of A_il
        // std::cout << "A_il offset: " << A_il[0].first << " A_il size: " << A_il[0].second << std::endl;
        // print size of A_il_j
        // std::cout << A_il_j.size() << std::endl;

        auto B_lj = matrix_slicing(N, K, l*k, l*k + k - 1, j*n, j*n + n - 1);
        auto B_lj_i = matrix_further_slice_by_row(B_lj, i*n_1, i*n_1 + n_1 - 1);
        // print size of B_lj_i
        // std::cout << B_lj_i.size() << std::endl;

        // moving A_il(j) to GPU
        uint64_t row_count = 0;
        for (auto row : A_il_j){
          g.insert(
            sendrecv_t{
              .src = cpu,
              .dst = gpu_grid[physical_GPU_id],
              .src_mem = {mem_t{.offset = row.first, .size = row.second}},
              .dst_mem = {mem_t{.offset = virtual_GPU_memory_offset + j * size_of_A_il_j + row_count * row.second, .size = row.second}}
              },
              {construct_A}
          );
          row_count++;
          // std::cout << "A source offset: " << row.first << " A destination offset: " 
          //             << virtual_GPU_memory_offset + j * size_of_A_il_j + row_count * row.second << " Size: " << row.second << std::endl;
        }

        // moving B_lj(i) to GPU
        row_count = 0;
        for (auto row: B_lj_i){
          g.insert(
            sendrecv_t{
              .src = cpu,
              .dst = gpu_grid[physical_GPU_id],
              .src_mem = {mem_t{.offset = lhs_size + row.first, .size = row.second}},
              .dst_mem = {mem_t{.offset = virtual_GPU_memory_offset + size_of_A_il + i * size_of_B_lj_i + row_count * row.second, .size = row.second}}
              },
              {construct_B}
          );
          row_count++;
          // std::cout << "B offset: " << lhs_size + row.first << std::endl;
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
  uint64_t num_virtual_GPUs_per_physical = (num_virtual_GPUs % num_physical_GPUs == 0) ? std::floor(num_virtual_GPUs / num_physical_GPUs) 
                                              : std::floor(num_virtual_GPUs / num_physical_GPUs) + 1;
  std::vector<uint64_t> memory_offset_virtual_GPUs;
  uint64_t virtual_GPU_memory_size = std::floor(gpu_mem_size / num_virtual_GPUs_per_physical);
  for (auto i = 0; i < num_virtual_GPUs_per_physical; ++i){
    memory_offset_virtual_GPUs.push_back(virtual_GPU_memory_size * i);
    // print offset
    std::cout << "virtual gpu: " << i << " offset: " << virtual_GPU_memory_size * i << std::endl;
  }

  graph_t g;

  // for testing
  auto GPU_num = 1;
  auto l_tmp = GPU_num % 2;

  // create a vector of size p_1 * p_3 to store all the communications of A_il_j
  // each element in the vector is a vector of size p_2
  // std::vector<std::vector<ident_t>> A_il_j_comms(p_1, std::vector<ident_t>(p_3));

  std::vector<std::vector<ident_t>> A_il_j_comms;
  for (auto i = 0; i < p_1 * p_3 * p_2; ++i){
    A_il_j_comms.push_back(std::vector<ident_t>());
  }
  std::vector<ident_t> A_comm_requirements;

  // create a vector of size p_2 * p_3 to store all the communications of B_lj_i
  // each element in the vector is a vector of size p_1
  // std::vector<std::vector<ident_t>> B_lj_i_comms(p_2, std::vector<ident_t>(p_3));

  std::vector<std::vector<ident_t>> B_lj_i_comms;
  for (auto i = 0; i < p_2 * p_3 * p_1; ++i){
    B_lj_i_comms.push_back(std::vector<ident_t>());
  }

  // create a vector of size p_1 * p_2 to store all the communications of C_ij_l
  // each element in the vector is a vector of size p_3
  // std::vector<std::vector<ident_t>> C_ij_l_comms(p_1, std::vector<ident_t>(p_2));

  std::vector<std::vector<ident_t>> C_ij_l_comms;
  for (auto i = 0; i < p_1 * p_2 * p_3; ++i){
    C_ij_l_comms.push_back(std::vector<ident_t>());
  }

  // This stores the compute requirement for each processor
  // each processor should only have one compute requirement
  std::vector<ident_t> compute_requirements;
  for (auto i = 0; i < p_1 * p_2 * p_3; ++i){
    compute_requirements.push_back(ident_t{});
  }

  std::vector<std::vector<ident_t>> aggregate_requirements;
  for (auto i = 0; i < p_1 * p_2 * p_3; ++i){
    aggregate_requirements.push_back(std::vector<ident_t>());
  }

    // This serves as the reference for the original data on CPU
    // ident_t move_to_host_before = g.insert(
    //       sendrecv_t{
    //         .src = gpu_grid[0],
    //         .dst = cpu,
    //         .src_mem = {mem_t{.offset = 512, .size = size_of_A_il_j}},
    //         .dst_mem = {mem_t{.offset = 512, .size = size_of_A_il_j}}
    //       });
    //     ident_t print_input_1_before = g.insert(apply_t{.loc = cpu, .read_mems = {mem_t{.offset = 512, .size = size_of_A_il_j}}, 
    //                               .write_mems = {}, 
    //                               .op = gen_print({m,k_2}) }, {move_to_host_before});

  // -------------- Let's start to construct the graph ------------------
  // Step 1: create the communication for A_il_j and B_lj_i
  // for processor (i, j ,l), it has A_il_j and we need to get A_il from all the A_il_j from other GPUs with the same i, l but different j
  // for processor (i, j ,l), it has B_lj_i and we need to get B_lj from all the B_lj_i from other GPUs with the same l, j but different i
  for (auto i = 0; i < p_1; ++i){
    for (auto j = 0; j < p_2; ++j){
      for (auto l = 0; l < p_3; ++l){
        auto virtual_GPU_id = i * p_2 * p_3 + j * p_3 + l;
        auto physical_GPU_id = std::floor(virtual_GPU_id / num_virtual_GPUs_per_physical);
        auto virtual_GPU_id_in_physical = virtual_GPU_id % num_virtual_GPUs_per_physical;
        auto virtual_GPU_memory_offset = memory_offset_virtual_GPUs[virtual_GPU_id_in_physical];

        // keep track of the offset for each tensor
        auto my_offset_start = virtual_GPU_memory_offset;
        auto my_offset_A_il_j = virtual_GPU_memory_offset + j * size_of_A_il_j;
        auto my_offset_B_lj_i = virtual_GPU_memory_offset + size_of_A_il + i * size_of_B_lj_i;
        auto my_offset_D_ij_l = virtual_GPU_memory_offset + size_of_A_il + size_of_B_lj;
        auto my_offset_D_ij_l_r = my_offset_D_ij_l + l * size_of_C_ij_l;
        auto my_offset_C_ij_l = my_offset_D_ij_l + size_of_D_ij_l * 2;

        // every processor needs to send its A_il_j to the processor p_2 - 1 times (no need to send to itself) 
        // every processor needs to send its B_lj_i to the processor p_1 - 1 times (no need to send to itself)
        
        // get A_il from all the A_il_j from other GPUs with the same i, l but different j
        for (auto j_2 = 0; j_2 < p_2; ++j_2){
          if (j_2 == j){
            continue;
          }
          auto other_virtual_GPU_id = i * p_2 * p_3 + j_2 * p_3 + l;
          auto other_physical_GPU_id = std::floor(other_virtual_GPU_id / num_virtual_GPUs_per_physical);
          // the starting offset for the other virtual GPU is different but the relative offset is the same
          auto other_virtual_GPU_memory_offset = memory_offset_virtual_GPUs[other_virtual_GPU_id % num_virtual_GPUs_per_physical] 
                                                                  + j * size_of_A_il_j;
          
          auto my_offset = virtual_GPU_memory_offset + j_2 * size_of_A_il_j;
          auto other_GPU_src = memory_offset_virtual_GPUs[other_virtual_GPU_id % num_virtual_GPUs_per_physical] 
                                                                  + j_2 * size_of_A_il_j;
          
          ident_t my_move_A;

          if (other_physical_GPU_id == physical_GPU_id){
            // if the other GPU is on the same physical GPU, we can just copy the data
            // insert a gen_gpu_move kernel
            my_move_A = g.insert(apply_t{ .loc = gpu_grid[physical_GPU_id],
                                .read_mems = {mem_t{.offset = my_offset_A_il_j, .size = size_of_A_il_j}}, 
                                .write_mems = {mem_t{.offset = other_virtual_GPU_memory_offset, .size = size_of_A_il_j}}, 
                                .op = gen_gpu_move(size_of_A_il_j) });
            // print src offset, dst offset and size
            std::cout << "Step 1 (Transfer A same GPU) src offset: " << my_offset_A_il_j << " dst offset: " << other_virtual_GPU_memory_offset 
                        << " size: " << size_of_A_il_j << " Src GPU: " << virtual_GPU_id << " Dst GPU: " 
                        << other_virtual_GPU_id << " command number: " << my_move_A << std::endl;
          }
          else {
            // if the other GPU is on a different physical GPU, we need to send the data
            // insert a sendrecv command
            my_move_A = g.insert(
              sendrecv_t{
                .src = gpu_grid[physical_GPU_id],
                .dst = gpu_grid[other_physical_GPU_id],
                .src_mem = {mem_t{.offset = my_offset_A_il_j, .size = size_of_A_il_j}},
                .dst_mem = {mem_t{.offset = other_virtual_GPU_memory_offset, .size = size_of_A_il_j}}
                }
              );
              // print src offset, dst offset and size
              std::cout << "Step 1 (Transfer A diff GPU) src offset: " << my_offset_A_il_j << " dst offset: " << other_virtual_GPU_memory_offset 
                          << " size: " << size_of_A_il_j << " Src GPU: " << virtual_GPU_id << " Dst GPU: " 
                          << other_virtual_GPU_id << " command number: " << my_move_A << std::endl;
          }

          // add the communication to the vector
          A_il_j_comms[other_virtual_GPU_id].push_back(my_move_A);
          A_comm_requirements.push_back(my_move_A);
        }

        // get B_lj from all the B_lj_i from other GPUs with the same l, j but different i
        for (auto i_2 = 0; i_2 < p_1; ++i_2){
          if (i_2 == i){
            continue;
          }
          auto other_virtual_GPU_id = i_2 * p_2 * p_3 + j * p_3 + l;
          auto other_physical_GPU_id = std::floor(other_virtual_GPU_id / num_virtual_GPUs_per_physical);
          auto other_virtual_GPU_memory_offset = memory_offset_virtual_GPUs[other_virtual_GPU_id % num_virtual_GPUs_per_physical] 
                                                                  + size_of_A_il + i * size_of_B_lj_i;

          auto my_offset = virtual_GPU_memory_offset + size_of_A_il + i_2 * size_of_B_lj_i;
          auto other_GPU_src = memory_offset_virtual_GPUs[other_virtual_GPU_id % num_virtual_GPUs_per_physical] 
                                                                  + size_of_A_il + i_2 * size_of_B_lj_i;

          ident_t my_move_B;
          if (other_physical_GPU_id == physical_GPU_id){
            // if the other GPU is on the same physical GPU, we can just copy the data
            // insert a gen_gpu_move kernel
            my_move_B = g.insert(apply_t{ .loc = gpu_grid[physical_GPU_id],
                                .read_mems = {mem_t{.offset = my_offset_B_lj_i, .size = size_of_B_lj_i}}, 
                                .write_mems = {mem_t{.offset = other_virtual_GPU_memory_offset, .size = size_of_B_lj_i}}, 
                                .op = gen_gpu_move(size_of_B_lj_i) });
            // print src and dst offset and size
            std::cout << "Step 1 (Transfer B same GPU) src offset: " << my_offset_B_lj_i << " dst offset: " << other_virtual_GPU_memory_offset 
                        << " size: " << size_of_B_lj_i << " Src GPU: " << virtual_GPU_id << " Dst GPU: " << other_virtual_GPU_id
                        <<  " command number: " << my_move_B << std::endl;
          }
          else {
            // if the other GPU is on a different physical GPU, we need to send the data
            // insert a sendrecv command
            my_move_B = g.insert(
              sendrecv_t{
                .src = gpu_grid[physical_GPU_id],
                .dst = gpu_grid[other_physical_GPU_id],
                .src_mem = {mem_t{.offset = my_offset_B_lj_i, .size = size_of_B_lj_i}},
                .dst_mem = {mem_t{.offset = other_virtual_GPU_memory_offset, .size = size_of_B_lj_i}}
                }
            );
            // print src and dst offset and size
            std::cout << "Step 1 (Transfer B diff GPU) src offset: " << my_offset_B_lj_i << " dst offset: " << other_virtual_GPU_memory_offset 
                        << " size: " << size_of_B_lj_i << " Src GPU: " << virtual_GPU_id << " Dst GPU: " 
                        << other_virtual_GPU_id << " command number: " << my_move_B << std::endl;
          }

          // add the communication to the vector
          B_lj_i_comms[other_virtual_GPU_id].push_back(my_move_B);
        }

      }}}
  
  // Checking Step 1: Checking if the communication is correct
  // check A_il_j communication
  ident_t move_to_host_1 = g.insert(
          sendrecv_t{
            .src = gpu_grid[0],
            .dst = cpu,
            .src_mem = {mem_t{.offset = 0, .size = size_of_A_il}},
            .dst_mem = {mem_t{.offset = 0, .size = size_of_A_il}}
          }, {A_il_j_comms[0]});
        ident_t print_input_1 = g.insert(apply_t{.loc = cpu, .read_mems = {mem_t{.offset = 0, .size = size_of_A_il}}, 
                                  .write_mems = {}, 
                                  .op = gen_print({m,k}) }, {move_to_host_1});
  // check B_lj_i communication
  ident_t move_to_host_2 = g.insert(
          sendrecv_t{
            .src = gpu_grid[0],
            .dst = cpu,
            .src_mem = {mem_t{.offset = size_of_A_il_j, .size = size_of_A_il}},
            .dst_mem = {mem_t{.offset = size_of_A_il_j, .size = size_of_A_il}}
          }, {A_il_j_comms[0]});
        ident_t print_input_2 = g.insert(apply_t{.loc = cpu, .read_mems = {mem_t{.offset = size_of_A_il_j, .size = size_of_B_lj}}, 
                                  .write_mems = {}, 
                                  .op = gen_print({k,n}) }, {move_to_host_2});

  
  
  
  // -------------- Step 2: Compute D_il_j = A_il * B_lj ------------------
  for (auto i = 0; i < p_1; ++i){
    for (auto j = 0; j < p_2; ++j){
      for (auto l = 0; l < p_3; ++l){
        auto virtual_GPU_id = i * p_2 * p_3 + j * p_3 + l;
        auto physical_GPU_id = std::floor(virtual_GPU_id / num_virtual_GPUs_per_physical);
        auto virtual_GPU_id_in_physical = virtual_GPU_id % num_virtual_GPUs_per_physical;
        auto virtual_GPU_memory_offset = memory_offset_virtual_GPUs[virtual_GPU_id_in_physical];

        // keep track of the offset for each tensor
        auto my_offset_start = virtual_GPU_memory_offset;
        auto my_offset_A_il_j = virtual_GPU_memory_offset + j * size_of_A_il_j;
        auto my_offset_B_lj_i = virtual_GPU_memory_offset + size_of_A_il + i * size_of_B_lj_i;
        auto my_offset_D_ij_l = virtual_GPU_memory_offset + size_of_A_il + size_of_B_lj;
        auto my_offset_D_ij_l_r = my_offset_D_ij_l + l * size_of_C_ij_l;
        auto my_offset_C_ij_l = my_offset_D_ij_l + size_of_D_ij_l * 2;


        // after the communications are done, perform the computation A_il * B_lj = D_ij^l
        // my_requirement is a vector with all elements from A_il_j_comms[i * p_1 + l * p_3] and B_lj_i_comms[j * p_2 + l * p_3]
        // print the size of A_il_j_comms[i * p_3 + l]
        // std::cout << "size of A_il_j_comms[i * p_3 + l] is " << A_il_j_comms[i * p_3 + l].size() << std::endl;
        // std::cout << "size of B_lj_i_comms[j * p_3 + l] is " << B_lj_i_comms[j * p_3 + l].size() << std::endl;
        std::vector<ident_t> my_requirement;
        for (auto A_comm: A_il_j_comms[virtual_GPU_id]){
          my_requirement.push_back(A_comm);
        }
        for (auto B_comm: B_lj_i_comms[virtual_GPU_id]){
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
        // print offsets of read_mems (my_offset_start and my_offset_start + size_of_A_il) and write_mems (my_offset_start + size_of_A_il + size_of_B_lj)
        // std::cout << " input offset 1: " << my_offset_start<< " input offset 2: " << my_offset_start + size_of_A_il << " output offset: " 
        //   << my_offset_start + size_of_A_il + size_of_B_lj << " size: " << size_of_D_ij_l << std::endl;
        compute_requirements[virtual_GPU_id] = my_compute;
        // print my input and output offsets and command number
        std::cout << "Step 2: Input 1: " << my_offset_start << " Input 2: " << my_offset_start + size_of_A_il << " Output: " 
          << my_offset_start + size_of_A_il + size_of_B_lj << " Src GPU: " << virtual_GPU_id <<  " command number: " << my_compute << std::endl;
          // print all the requirements
          for (auto req: my_requirement){
            std::cout << "Requirement: " << req << std::endl;
          }

      }}}

  // Checking step 2
  // do a test to see if the computation is correct
  ident_t move_to_host_compute = g.insert(
    sendrecv_t{
      .src = gpu_grid[0],
      .dst = cpu,
      .src_mem = {mem_t{.offset = size_of_A_il + size_of_B_lj, .size = size_of_D_ij_l}},
      .dst_mem = {mem_t{.offset = size_of_A_il + size_of_B_lj, .size = size_of_D_ij_l}}
    }, {compute_requirements[0]});
  ident_t print_compute = g.insert(apply_t{.loc = cpu, .read_mems = {mem_t{.offset = size_of_A_il + size_of_B_lj, .size = size_of_D_ij_l}}, .write_mems = {}, 
                            .op = gen_print({m,n}) }, {move_to_host_compute});

  
  
  
  // -------------- Step 3: perform the communications D_ij^l_r -------------------
  for (auto i = 0; i < p_1; ++i){
    for (auto j = 0; j < p_2; ++j){
      for (auto l = 0; l < p_3; ++l){
        auto virtual_GPU_id = i * p_2 * p_3 + j * p_3 + l;
        auto physical_GPU_id = std::floor(virtual_GPU_id / num_virtual_GPUs_per_physical);
        auto virtual_GPU_id_in_physical = virtual_GPU_id % num_virtual_GPUs_per_physical;
        auto virtual_GPU_memory_offset = memory_offset_virtual_GPUs[virtual_GPU_id_in_physical];

        // keep track of the offset for each tensor
        auto my_offset_start = virtual_GPU_memory_offset;
        auto my_offset_A_il_j = virtual_GPU_memory_offset + j * size_of_A_il_j;
        auto my_offset_B_lj_i = virtual_GPU_memory_offset + size_of_A_il + i * size_of_B_lj_i;
        auto my_offset_D_ij_l = virtual_GPU_memory_offset + size_of_A_il + size_of_B_lj;
        auto my_offset_D_ij_l_r = my_offset_D_ij_l + l * size_of_C_ij_l;
        auto my_offset_C_ij_l = my_offset_D_ij_l + size_of_D_ij_l * 2;

        // get D_ij_l_r from all the D_ij_l_r from other GPUs with the same i, j but different l
        // originally D_ij_l_r is a m * n_3 = m * (n / p_3) matrix
        // but if we slice it across rows we can have D_ij_l_r = (m / p_3) * n matrix 
        // auto D_ij_l_r = matrix_slicing(m, n, l * m / p_3, (l + 1) * m / p_3 - 1, 0, n - 1);

        for (auto l_2 = 0; l_2 < p_3; ++l_2){
          auto other_virtual_GPU_id = i * p_2 * p_3 + j * p_3 + l_2;
          auto other_physical_GPU_id = std::floor(other_virtual_GPU_id / num_virtual_GPUs_per_physical);
          auto other_virtual_GPU_memory_offset = memory_offset_virtual_GPUs[other_virtual_GPU_id % num_virtual_GPUs_per_physical] 
                                                                  + size_of_A_il + size_of_B_lj + size_of_D_ij_l + l * size_of_C_ij_l;

          auto my_offset = my_offset_D_ij_l_r + size_of_D_ij_l + l_2 * size_of_C_ij_l;
          auto other_GPU_src = memory_offset_virtual_GPUs[other_virtual_GPU_id % num_virtual_GPUs_per_physical] + size_of_A_il 
                                + size_of_B_lj + size_of_D_ij_l + l_2 * size_of_C_ij_l;

          if (l_2 == l){
            C_ij_l_comms[virtual_GPU_id].push_back(compute_requirements[virtual_GPU_id]);
            continue;
          }
          ident_t my_move_D;
          // if the other GPU is on the same physical GPU, we can just copy the data
          // insert a gen_gpu_move kernel
          if (other_physical_GPU_id == physical_GPU_id){
            my_move_D = g.insert(apply_t{ .loc = gpu_grid[physical_GPU_id],
                                .read_mems = {mem_t{.offset = my_offset_D_ij_l_r, .size = size_of_C_ij_l}}, 
                                .write_mems = {mem_t{.offset = other_virtual_GPU_memory_offset, .size = size_of_C_ij_l}}, 
                                .op = gen_gpu_move(size_of_C_ij_l)}, 
                                {compute_requirements[virtual_GPU_id]});

            // print read offset, write offset
            std::cout << "Step 3: read offset (same GPU): " << my_offset_D_ij_l_r << " write offset: " << other_virtual_GPU_memory_offset << " size: " 
              << size_of_C_ij_l << " Src GPU: " << virtual_GPU_id << " Dst GPU: " << other_virtual_GPU_id
              << " command number: " << my_move_D << std::endl;
          }
          else {
            // if the other GPU is on a different physical GPU, we need to send the data
            // insert a sendrecv command
            my_move_D = g.insert(
              sendrecv_t{
                .src = gpu_grid[physical_GPU_id],
                .dst = gpu_grid[other_physical_GPU_id],
                .src_mem = {mem_t{.offset = my_offset_D_ij_l_r, .size = size_of_C_ij_l}},
                .dst_mem = {mem_t{.offset = other_virtual_GPU_memory_offset, .size = size_of_C_ij_l}}
                }, {compute_requirements[virtual_GPU_id]}
            );
            std::cout << "Step 3: read offset: " << my_offset_D_ij_l_r << " write offset: " << other_virtual_GPU_memory_offset 
              << " size: " << size_of_C_ij_l << " Src GPU: " << virtual_GPU_id <<  " Dest GPU: " << other_virtual_GPU_id 
              << " command number: " << my_move_D << std::endl;
          }

          C_ij_l_comms[other_virtual_GPU_id].push_back(my_move_D);

          // print my compute requirement
          std::cout << "My compute requirement: " << compute_requirements[virtual_GPU_id] << std::endl;

          // print my_offset_D_ij_l_r, other_virtual_GPU_memory_offset, size_of_D_ij_l and physical_GPU_id & other_physical_GPU_id
          // std::cout << "Src offset: " << my_offset_D_ij_l_r << " Dst offset: " << other_virtual_GPU_memory_offset << " Size: " 
          //     << size_of_D_ij_l << " Src GPU: " << physical_GPU_id << " Dst GPU: " << other_physical_GPU_id << std::endl;
        }

      }}}

  // Checking step 3
  // do a test to see if the moves are correct
  uint64_t D_ij_l_r_offset_test = memory_offset_virtual_GPUs[GPU_num] + size_of_A_il + size_of_B_lj + size_of_D_ij_l;
  ident_t move_to_host_D_ij_l_r = g.insert(
    sendrecv_t{
      .src = gpu_grid[0],
      .dst = cpu,
      .src_mem = {mem_t{.offset = D_ij_l_r_offset_test, .size = size_of_D_ij_l}},
      .dst_mem = {mem_t{.offset = 0, .size = size_of_D_ij_l}}
    }, {C_ij_l_comms[0]});
  ident_t print_D_ij_l_r_move = g.insert(apply_t{.loc = cpu, .read_mems = {mem_t{.offset = 0, .size = size_of_D_ij_l}}, .write_mems = {}, 
                            .op = gen_print({m,n}) }, {move_to_host_D_ij_l_r});
  std::cout << "Step 3 test: starting offset: " << D_ij_l_r_offset_test << " size: " << size_of_D_ij_l << std::endl;

  // auto starting_offset_3_2 = memory_offset_virtual_GPUs[GPU_num]+ size_of_A_il + size_of_B_lj 
  //                                         + size_of_D_ij_l * 2;

  // // another check for the before hand final result address value
  // // print my starting offset
  // // std::cout << "Step 3 test 2: starting offset: " << starting_offset_3_2 << std::endl;
  // // ident_t move_to_host_agg_2 = g.insert(
  // //   sendrecv_t{
  // //     .src = gpu_grid[0],
  // //     .dst = cpu,
  // //     .src_mem = {mem_t{.offset = starting_offset_3_2, .size = size_of_D_ij_l}},
  // //     .dst_mem = {mem_t{.offset = 0, .size = size_of_D_ij_l}}
  // //   }, {print_D_ij_l_r_move});
  // // uint64_t num_row_2 = std::floor(m / p_3);
  // // ident_t print_agg_2 = g.insert(apply_t{.loc = cpu, .read_mems = {mem_t{.offset = 0, .size = size_of_D_ij_l}}, .write_mems = {}, 
  // //                           .op = gen_print({m, n}) }, {move_to_host_agg_2});

  
  
  
  // -------------- Step 4: Aggregate all the communications -----------------------
  for (auto i = 0; i < p_1; ++i){
    for (auto j = 0; j < p_2; ++j){
      for (auto l = 0; l < p_3; ++l){
        auto virtual_GPU_id = i * p_2 * p_3 + j * p_3 + l;
        auto physical_GPU_id = std::floor(virtual_GPU_id / num_virtual_GPUs_per_physical);
        auto virtual_GPU_id_in_physical = virtual_GPU_id % num_virtual_GPUs_per_physical;
        auto virtual_GPU_memory_offset = memory_offset_virtual_GPUs[virtual_GPU_id_in_physical];

        // keep track of the offset for each tensor
        auto my_offset_start = virtual_GPU_memory_offset;
        auto my_offset_A_il_j = virtual_GPU_memory_offset + j * size_of_A_il_j;
        auto my_offset_B_lj_i = virtual_GPU_memory_offset + size_of_A_il + i * size_of_B_lj_i;
        auto my_offset_D_ij_l = virtual_GPU_memory_offset + size_of_A_il + size_of_B_lj;
        auto my_offset_D_ij_l_r = my_offset_D_ij_l + l * size_of_C_ij_l;
        auto my_offset_C_ij_l = my_offset_D_ij_l + size_of_D_ij_l * 2;

        uint64_t num_row = std::floor(m / p_3);

        // print my_virtual_GPU_id, my_physical_GPU_id
        // std::cout << "my_virtual_GPU_id: " << virtual_GPU_id << " my_physical_GPU_id: " << physical_GPU_id << std::endl;

        // Finally aggregate all the communications
        for (auto l_2 = 0; l_2 < p_3; ++l_2){
          // do multiple matrix adds
          // my_requirement_C is a vector with all elements from C_ij_l_comms[i * p_1 + j * p_2]
          // std::cout << "input offset: " << my_offset_D_ij_l + size_of_D_ij_l + l_2 * size_of_C_ij_l << std::endl;
          // std::cout << "output offset: " << my_offset_C_ij_l << std::endl;
          // std::cout << "size: " << size_of_C_ij_l << std::endl;
          if (l_2 == l){
            ident_t my_agg = g.insert(
            apply_t {
              .loc = gpu_grid[physical_GPU_id],
              .read_mems = {mem_t{.offset = my_offset_D_ij_l_r, .size = size_of_C_ij_l}, 
                                {mem_t{.offset = my_offset_C_ij_l, .size = size_of_C_ij_l}}},
              .write_mems = {mem_t{.offset = my_offset_C_ij_l, .size = size_of_C_ij_l}},
              .op = gen_gpu_matadd(num_row, n)}, 
              C_ij_l_comms[virtual_GPU_id]);

              // print aggregate input offsets and output offsets
              std::cout << "Step 4: input offset 1 (same Dev): " << my_offset_D_ij_l_r << " input offset 2: " << my_offset_C_ij_l << 
                " output offset: " << my_offset_C_ij_l << " Src GPU: " << virtual_GPU_id << std::endl;
              // print requirements
              for (auto requirement: C_ij_l_comms[virtual_GPU_id]){
                std::cout << "requirement: " << requirement << std::endl;
              }
            aggregate_requirements[virtual_GPU_id].push_back(my_agg);
          }
          else{
            auto my_input = mem_t{.offset = my_offset_D_ij_l + size_of_D_ij_l + l_2 * size_of_C_ij_l, .size = size_of_C_ij_l};
            auto my_output = mem_t{.offset = my_offset_C_ij_l, .size = size_of_C_ij_l};
            ident_t my_agg = g.insert(
              apply_t {
                .loc = gpu_grid[physical_GPU_id],
                .read_mems = {my_input, my_output},
                .write_mems = {my_output},
                .op = gen_gpu_matadd(num_row, n)}, 
                C_ij_l_comms[virtual_GPU_id]);

            // print aggregate input offsets and output offsets
              std::cout << "Step 4: input offset 1 (diff Dev): " << my_offset_D_ij_l + size_of_D_ij_l + l_2 * size_of_C_ij_l 
                << " input offset 2: " << my_offset_C_ij_l << " output offset: "  << my_offset_C_ij_l
                << " Src GPU: " << virtual_GPU_id << std::endl;
              // print requirements
              for (auto requirement: C_ij_l_comms[virtual_GPU_id]){
                std::cout << "requirement: " << requirement << std::endl;
              }
            
            aggregate_requirements[virtual_GPU_id].push_back(my_agg);
          }
          
        }
      }}}

  // Checking step 4
  auto starting_offset_4 = memory_offset_virtual_GPUs[GPU_num]+ size_of_A_il + size_of_B_lj 
                                          + size_of_D_ij_l * 2;
  // print my starting offset
  std::cout << "Step 4 test: starting offset: " << starting_offset_4 << std::endl;
  ident_t move_to_host_agg = g.insert(
    sendrecv_t{
      .src = gpu_grid[0],
      .dst = cpu,
      .src_mem = {mem_t{.offset = starting_offset_4, .size = size_of_C_ij_l}},
      .dst_mem = {mem_t{.offset = 0, .size = size_of_C_ij_l}}
    }, {aggregate_requirements[0]});
  uint64_t num_row = std::floor(m / p_3);
  ident_t print_agg = g.insert(apply_t{.loc = cpu, .read_mems = {mem_t{.offset = 0, .size = size_of_C_ij_l}}, .write_mems = {}, 
                            .op = gen_print({num_row, n}) }, {move_to_host_agg});

  
  
  // at the end we return the graph
  return g;
}