// Udacity HW 4
// Radix Sorting

#include <thrust/host_vector.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>
#include "utils.h"

/* Red Eye Removal
   ===============

   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */

const unsigned int BIT_PER_CYCLE = 2;
const unsigned int BIT_MASK = (1 << BIT_PER_CYCLE) - 1;
const unsigned int INVALID_BIN = 1 << BIT_PER_CYCLE;

#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)

__device__ int block_scan(int* const arr, int tid, int block_size, int* sum) {
  int offset = 1;
  for (int d = block_size >> 1; d > 0; d >>= 1) {
    if (tid < d) {
      int ai = offset * (2 * tid + 1) - 1;
      int bi = offset * (2 * tid + 2) - 1;
      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);
      arr[bi] += arr[ai];
    }
    __syncthreads();
    offset *= 2;
  }
  if (tid == 0) {
    int last_idx = block_size - 1 + CONFLICT_FREE_OFFSET(block_size - 1);
    *sum = arr[last_idx];
    arr[last_idx] = 0;
  }

  __syncthreads();
  for (int d = 1; d < block_size; d <<= 1) {
    offset >>= 1;
    if (tid < d) {
      int ai = offset * (2 * tid + 1) - 1;
      int bi = offset * (2 * tid + 2) - 1;
      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);
      unsigned int v = arr[ai];
      arr[ai] = arr[bi];
      arr[bi] += v;
    }
    __syncthreads();
  }
}

template <size_t BLOCK_SIZE>
__global__ void block_radix_sort_one_pass(
    unsigned int* const d_inputVals,
    unsigned int* const d_inputPos,
    unsigned int* const d_outputVals,
    unsigned int* const d_outputPos,
    const size_t numElems,
    int* const d_blockSums,
    const size_t alignedNumBins,
    int shift) {
  __shared__ int temp[BLOCK_SIZE * 2 + 32];
  __shared__ int block_sum[1 << BIT_PER_CYCLE];
  int tid = threadIdx.x;
  int leaf_num = BLOCK_SIZE * 2;
  int block_offset = blockIdx.x * leaf_num;
  unsigned int val_a, val_b, pos_a, pos_b;
  int ai = tid;
  int bi = tid + BLOCK_SIZE;
  int bin_idx_a, bin_idx_b;
  if (ai + block_offset < numElems) {
    val_a = d_inputVals[ai + block_offset];
    pos_a = d_inputPos[ai + block_offset];
  }
  if (bi + block_offset < numElems) {
    val_b = d_inputVals[bi + block_offset];
    pos_b = d_inputPos[bi + block_offset];
  }
  int bin_ai =
      ai + block_offset < numElems ? (val_a >> shift) & BIT_MASK : INVALID_BIN;
  int bin_bi =
      bi + block_offset < numElems ? (val_b >> shift) & BIT_MASK : INVALID_BIN;
  int bank_offset_ai = CONFLICT_FREE_OFFSET(ai);
  int bank_offset_bi = CONFLICT_FREE_OFFSET(bi);
  int* blockSums = &d_blockSums[blockIdx.x];
  int local_ai_idx = 0, local_bi_idx = 0;

  // block scan
  for (int bin = 0; bin <= BIT_MASK; bin++) {
    temp[ai + bank_offset_ai] = bin_ai == bin ? 1 : 0;
    temp[bi + bank_offset_bi] = bin_bi == bin ? 1 : 0;
    __syncthreads();
    block_scan(temp, tid, leaf_num, &block_sum[bin]);
    if (tid == 0) {
      *blockSums = block_sum[bin];
      blockSums += alignedNumBins;
    }
    if (bin < bin_ai)
      local_ai_idx += block_sum[bin];
    if (bin < bin_bi)
      local_bi_idx += block_sum[bin];
    if (bin_ai == bin)
      bin_idx_a = temp[ai + bank_offset_ai];
    if (bin_bi == bin)
      bin_idx_b = temp[bi + bank_offset_bi];
  }
  __syncthreads();

  local_ai_idx += bin_idx_a;
  local_bi_idx += bin_idx_b;

  if (ai + block_offset < numElems) {
    d_outputVals[local_ai_idx + block_offset] = val_a;
    d_outputPos[local_ai_idx + block_offset] = pos_a;
  }
  if (bi + block_offset < numElems) {
    d_outputVals[local_bi_idx + block_offset] = val_b;
    d_outputPos[local_bi_idx + block_offset] = pos_b;
  }
}

template <size_t BLOCK_SIZE>
__global__ void map_global_position(
    int* const d_blockSums,
    int* const d_globalPositions,
    const size_t alignedNumBlocks,
    const size_t numBlocks) {
  int tid = threadIdx.x;
  int* blockSums = d_blockSums;
  int* globalPositions = d_globalPositions;
  __shared__ int temp[BLOCK_SIZE * 2 + 32];
  __shared__ int sum[2];
  int ai = tid;
  int bi = tid + BLOCK_SIZE;
  int leaf_num = BLOCK_SIZE * 2;
  int bank_offset_ai = CONFLICT_FREE_OFFSET(ai);
  int bank_offset_bi = CONFLICT_FREE_OFFSET(bi);
  if (tid == 0) {
    sum[1] = 0;
  }
  __syncthreads();
  for (int i = 0; i <= BIT_MASK; i++) {
    temp[ai + bank_offset_ai] = ai < numBlocks ? blockSums[ai] : 0;
    temp[bi + bank_offset_bi] = bi < numBlocks ? blockSums[bi] : 0;
    __syncthreads();
    // if (tid == 0) {
    //   for (int j = 0; j < leaf_num; j++)
    //     printf("%d ", temp[j + CONFLICT_FREE_OFFSET(j)]);
    //   printf("\n");
    // }
    block_scan(temp, tid, leaf_num, &sum[0]);
    // if (tid == 0) {
    //   printf("bin %d %d\n", i, sum[0]);
    // }
    if (ai < numBlocks)
      globalPositions[ai] = temp[ai + bank_offset_ai] + sum[1];
    if (bi < numBlocks)
      globalPositions[bi] = temp[bi + bank_offset_bi] + sum[1];
    if (tid == 0) {
      sum[1] += sum[0];
    }
    __syncthreads();
    blockSums += alignedNumBlocks;
    globalPositions += alignedNumBlocks;
  }
}

__global__ void scatter(
    unsigned int* const d_inputVals,
    unsigned int* const d_inputPos,
    unsigned int* const d_outputVals,
    unsigned int* const d_outputPos,
    const size_t numElems,
    int* const d_blockSums,
    int* const d_globalPositions,
    const size_t alignedNumBlocks,
    const int shift) {
  int tid = threadIdx.x;
  int* blockSums = d_blockSums;
  int* globalPositions = d_globalPositions;
  __shared__ int sums[1 << BIT_PER_CYCLE];
  __shared__ int positions[1 << BIT_PER_CYCLE];
  int block_offset = blockIdx.x * blockDim.x;
  int idx = block_offset + tid;
  int val, pos, bin = INVALID_BIN;
  if (tid == 0) {
#pragma unroll
    for (int i = 0; i < (1 << BIT_PER_CYCLE); i++) {
      sums[i] = blockSums[blockIdx.x];
      positions[i] = globalPositions[blockIdx.x];
      blockSums += alignedNumBlocks;
      globalPositions += alignedNumBlocks;
    }
  }
  __syncthreads();

  if (idx < numElems) {
    val = d_inputVals[idx];
    pos = d_inputPos[idx];
    bin = (val >> shift) & BIT_MASK;
    int p = positions[bin];
    int bin_start = 0;
    for (int i = 0; i < bin; i++)
      bin_start += sums[i];
    int global_position = p + tid - bin_start;
    d_outputVals[global_position] = val;
    d_outputPos[global_position] = pos;
  }
}

void your_sort(
    unsigned int* const d_inputVals,
    unsigned int* const d_inputPos,
    unsigned int* const d_outputVals,
    unsigned int* const d_outputPos,
    const size_t numElems) {
  // std::cout << numElems << std::endl;
  const int BLOCK_SIZE = 256;
  const int MAP_BLOCK_SIZE = 256;
  const int NUM_PER_BLOCK = BLOCK_SIZE * 2;
  int grid_size = (numElems + NUM_PER_BLOCK - 1) / NUM_PER_BLOCK;
  int aligned_grid_size = (grid_size + 7) / 8 * 8;
  int* d_block_sums;
  int* d_global_positions;

  checkCudaErrors(cudaMalloc(
      &d_block_sums, sizeof(int) * (1 << BIT_PER_CYCLE) * aligned_grid_size));
  checkCudaErrors(cudaMalloc(
      &d_global_positions,
      sizeof(int) * (1 << BIT_PER_CYCLE) * aligned_grid_size));
  for (int shift = 0; shift < sizeof(unsigned int) * 8;
       shift += BIT_PER_CYCLE) {
    block_radix_sort_one_pass<BLOCK_SIZE><<<grid_size, BLOCK_SIZE>>>(
        d_inputVals,
        d_inputPos,
        d_outputVals,
        d_outputPos,
        numElems,
        d_block_sums,
        aligned_grid_size,
        shift);
    map_global_position<MAP_BLOCK_SIZE><<<1, MAP_BLOCK_SIZE>>>(
        d_block_sums, d_global_positions, aligned_grid_size, grid_size);
    scatter<<<grid_size, NUM_PER_BLOCK>>>(
        d_outputVals,
        d_outputPos,
        d_inputVals,
        d_inputPos,
        numElems,
        d_block_sums,
        d_global_positions,
        aligned_grid_size,
        shift);
  }
  checkCudaErrors(cudaMemcpy(
      d_outputVals,
      d_inputVals,
      sizeof(unsigned int) * numElems,
      cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy(
      d_outputPos,
      d_inputPos,
      sizeof(unsigned int) * numElems,
      cudaMemcpyDeviceToDevice));
}
