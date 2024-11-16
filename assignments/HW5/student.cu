/* Udacity HW5
   Histogramming for Speed

   The goal of this assignment is compute a histogram
   as fast as possible.  We have simplified the problem as much as
   possible to allow you to focus solely on the histogramming algorithm.

   The input values that you need to histogram are already the exact
   bins that need to be updated.  This is unlike in HW3 where you needed
   to compute the range of the data and then do:
   bin = (val - valMin) / valRange to determine the bin.

   Here the bin is just:
   bin = val

   so the serial histogram calculation looks like:
   for (i = 0; i < numElems; ++i)
     histo[val[i]]++;

   That's it!  Your job is to make it run as fast as possible!

   The values are normally distributed - you may take
   advantage of this fact in your implementation.

*/

#include "utils.h"

static const int NUM_BINS = 1024;
static const int BLOCK_SIZE = 1024;
static const int WARP_SIZE = 32;
static const int WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;
// static const int BINS_PER_THREAD = 32;
static const int BINS_PER_THREAD = NUM_BINS / BLOCK_SIZE * WARP_SIZE;
static const int WRITES_PER_THREAD = BINS_PER_THREAD / WARP_SIZE;
int div_round_up(int num, int den) {
  return (num + den - 1) / den;
}

__global__ void calc_bins(
    const unsigned* const d_bin,
    unsigned* d_histo_temp,
    const size_t num_bins,
    const size_t reads_per_warp,
    const size_t total_values) {
  int warp = threadIdx.x / WARP_SIZE;
  int warp_idx = threadIdx.x % WARP_SIZE;
  int histo_start = warp * BINS_PER_THREAD;
  int histo_end = histo_start + BINS_PER_THREAD;
  int idx = blockIdx.x * (reads_per_warp * WARP_SIZE) + warp_idx;
  unsigned histo_bins[BINS_PER_THREAD];

  for (int i = 0; i < BINS_PER_THREAD; i++) {
    histo_bins[i] = 0;
  }

  for (int i = 0; i < reads_per_warp && idx < total_values; i++) {
    int bin = d_bin[idx];
    if (bin >= histo_start && bin < histo_end) {
      histo_bins[bin - histo_start]++;
    }

    idx += WARP_SIZE;
  }

  // Use Warp Shuffle to reduce bin values from all threads in a warp to a
  // single value
  for (int i = 0; i < BINS_PER_THREAD; i++) {
    for (int w = WARP_SIZE / 2; w > 0; w = w / 2) {
      histo_bins[i] += __shfl_down_sync(0xffffffff, histo_bins[i], w);
    }
  }

  // Re-broadcast so all threads have warp_0's values
  for (int i = 0; i < BINS_PER_THREAD; i++) {
    histo_bins[i] = __shfl_sync(0xffffffff, histo_bins[i], 0);
  }

  for (int i = 0; i < WRITES_PER_THREAD; i++) {
    unsigned warp_idx_itr = warp_idx + i * WARP_SIZE;
    d_histo_temp
        [blockIdx.x * (WARPS_PER_BLOCK * BINS_PER_THREAD) + histo_start +
         warp_idx_itr] = histo_bins[warp_idx_itr];
  }
}

__global__ void reduce_bins(
    const unsigned* const d_histo_bin,
    unsigned* d_histo_out,
    const size_t num_bins,
    const size_t total_values) {
  int idx = threadIdx.x;
  int bin_count = 0;

  for (int i = 0; i < total_values; i++) {
    bin_count += d_histo_bin[idx];
    idx += num_bins;
  }

  // this only works if blockDim.x == num_bins
  d_histo_out[threadIdx.x] = bin_count;

  // printf("bin %d = %d\n", threadIdx.x, bin_count);
}

void computeHistogram(
    const unsigned int* const d_vals, // INPUT
    unsigned int* const d_histo, // OUTPUT
    const unsigned int numBins,
    const unsigned int numElems) {
  // TODO Launch the yourHisto kernel

  // if you want to use/launch more than one kernel,
  // feel free

  cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1024 * 1024 * 1024);
  dim3 block_size(BLOCK_SIZE, 1, 1);

  // ALLOCS
  const size_t total_values_p2a = numElems;
  const int reads_per_warp_p2a = 2048;
  dim3 grid_size_p2a(
      div_round_up(
          total_values_p2a, block_size.x / WARP_SIZE * reads_per_warp_p2a),
      1,
      1);

  const size_t total_values_p2b = grid_size_p2a.x;
  dim3 grid_size_p2b(1, 1, 1);

  unsigned* d_histo_temp;
  checkCudaErrors(
      cudaMalloc(&d_histo_temp, sizeof(unsigned) * grid_size_p2a.x * numBins));

  float elapsed = 0;
  cudaEvent_t start, c_bin, r_bin;

  cudaEventCreate(&start);
  cudaEventCreate(&c_bin);
  cudaEventCreate(&r_bin);

  // Calc Histogram bins

  cudaEventRecord(start, 0);
  calc_bins<<<grid_size_p2a, block_size>>>(
      d_vals, d_histo_temp, numBins, reads_per_warp_p2a, total_values_p2a);
  cudaEventRecord(c_bin, 0);
  reduce_bins<<<grid_size_p2b, 1024>>>(
      d_histo_temp, d_histo, numBins, total_values_p2b);
  cudaEventRecord(r_bin, 0);
  cudaEventSynchronize(r_bin);

  cudaEventElapsedTime(&elapsed, start, c_bin);
  printf("calc_bins kernel runtime was %.4f ms\n", elapsed);

  cudaEventElapsedTime(&elapsed, c_bin, r_bin);
  printf("reduce_bins kernel runtime was %.4f ms\n", elapsed);

  cudaEventElapsedTime(&elapsed, start, r_bin);
  printf("total kernel runtime was %.4f ms\n", elapsed);

  cudaEventDestroy(start);
  cudaEventDestroy(c_bin);
  cudaEventDestroy(r_bin);

  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());
}
