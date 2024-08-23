/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.


  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include <float.h>
#include <random>
#include "utils.h"
#define WARP_SIZE 32
template <size_t blockSize>
__device__ void reduce_warp(volatile float2* const sdata, size_t tid) {
  if (blockSize >= 32) {
    sdata[tid].x = min(sdata[tid].x, sdata[tid + 16].x);
    sdata[tid].y = max(sdata[tid].y, sdata[tid + 16].y);
  }
  if (blockSize >= 16) {
    sdata[tid].x = min(sdata[tid].x, sdata[tid + 8].x);
    sdata[tid].y = max(sdata[tid].y, sdata[tid + 8].y);
  }
  if (blockSize >= 8) {
    sdata[tid].x = min(sdata[tid].x, sdata[tid + 4].x);
    sdata[tid].y = max(sdata[tid].y, sdata[tid + 4].y);
  }
  if (blockSize >= 4) {
    sdata[tid].x = min(sdata[tid].x, sdata[tid + 2].x);
    sdata[tid].y = max(sdata[tid].y, sdata[tid + 2].y);
  }
  if (blockSize >= 2) {
    sdata[tid].x = min(sdata[tid].x, sdata[tid + 1].x);
    sdata[tid].y = max(sdata[tid].y, sdata[tid + 1].y);
  }
}

template <size_t warpSize>
__device__ __forceinline__ float2
warpReduceMinMax(float min_val, float max_val) {
#pragma unroll
  for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
    min_val = min(__shfl_down_sync(0xffffffff, min_val, offset), min_val);
    max_val = max(__shfl_down_sync(0xffffffff, max_val, offset), max_val);
  }
  return make_float2(min_val, max_val);
}

template <size_t blockSize>
__device__ float2
block_reduce_min_max(const float min_val, const float max_val) {
  const int tid = threadIdx.x;
  const int wid = tid / WARP_SIZE;
  const int lid = tid % WARP_SIZE;

  __shared__ float2 shared[WARP_SIZE];
  float2 val = warpReduceMinMax<WARP_SIZE>(min_val, max_val);
  // __syncthreads();
  if (lid == 0) {
    shared[wid] = val;
  }
  __syncthreads();
  val = (wid == 0 && tid < blockSize / WARP_SIZE) ? shared[lid]
                                                  : make_float2(FLT_MAX, 0);
  if (wid == 0) {
    val = warpReduceMinMax<blockSize / WARP_SIZE>(val.x, val.y);
  }
  return val;
}

template <size_t blockSize>
__global__ void reduce_min_max_first_kernel(
    const float* const d_input,
    float2* const g_output,
    size_t num_p_thread,
    size_t total_num) {
  size_t tid = threadIdx.x;
  size_t idx = blockIdx.x * blockSize * num_p_thread + tid;
  float min_val = FLT_MAX, max_val = 0;

#pragma unroll
  for (int i = 0; i < num_p_thread; i++) {
    if (idx < total_num) {
      min_val = min(min_val, d_input[idx]);
      max_val = max(max_val, d_input[idx]);
      idx += blockSize;
    }
  }
  float2 val = block_reduce_min_max<blockSize>(min_val, max_val);
  if (tid == 0) {
    g_output[blockIdx.x] = val;
  }
}

template <size_t blockSize>
__global__ void reduce_min_max_second_kernel(
    const float2* const d_input,
    float2* const g_output,
    size_t num_p_thread,
    size_t total_num) {
  size_t tid = threadIdx.x;
  size_t idx = blockIdx.x * blockSize * num_p_thread + tid;
  float min_val = FLT_MAX, max_val = 0;

#pragma unroll
  for (int i = 0; i < num_p_thread; i++) {
    if (idx < total_num) {
      min_val = min(min_val, d_input[idx].x);
      max_val = max(max_val, d_input[idx].y);
      idx += blockSize;
    }
  }
  float2 val = block_reduce_min_max<blockSize>(min_val, max_val);
  if (tid == 0) {
    g_output[blockIdx.x] = val;
  }
}

std::pair<size_t, size_t> get_number_blocks(size_t n, size_t block_size) {
  int dev, sm_count;
  checkCudaErrors(cudaGetDevice(&dev));

  checkCudaErrors(
      cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev));
  int tpm;
  checkCudaErrors(cudaDeviceGetAttribute(
      &tpm, cudaDevAttrMaxThreadsPerMultiProcessor, dev));

  size_t num_blocks = std::min<size_t>(
      (n + block_size - 1) / block_size, sm_count * tpm / block_size);
  size_t thread_num = block_size * num_blocks;
  size_t num_p_thread = (n + thread_num - 1) / thread_num;
  return std::make_pair(num_blocks, num_p_thread);
}

template <size_t blockSize>
float2 reduce_min_max(const float* const d_input, size_t total_num) {
  auto p = get_number_blocks(total_num, blockSize);
  dim3 grid_size(p.first, 1, 1);
  dim3 block_size(blockSize, 1, 1);

  float2* h_output = (float2*)malloc(2 * sizeof(float2));
  float2 *d_output, *d_output2;
  checkCudaErrors(cudaMalloc(&d_output, sizeof(float2) * p.first));
  checkCudaErrors(cudaMalloc(&d_output2, sizeof(float2) * p.first));

  reduce_min_max_first_kernel<blockSize>
      <<<grid_size, block_size>>>(d_input, d_output, p.second, total_num);
  checkCudaErrors(cudaMemcpy(
      h_output, d_output, 2 * sizeof(float2), cudaMemcpyDeviceToHost));

  while (p.first > 1) {
    total_num = p.first;
    p = get_number_blocks(total_num, blockSize);
    grid_size = dim3(p.first, 1, 1);
    reduce_min_max_second_kernel<blockSize>
        <<<grid_size, block_size>>>(d_output, d_output2, p.second, total_num);
    std::swap(d_output, d_output2);
  }
  checkCudaErrors(
      cudaMemcpy(h_output, d_output, sizeof(float2), cudaMemcpyDeviceToHost));
  float2 res = h_output[0];
  free(h_output);
  checkCudaErrors(cudaFree(d_output));
  checkCudaErrors(cudaFree(d_output2));
  return res;
}

__global__ void histogram(
    const float* const d_input,
    size_t total_num,
    int reads_per_thread,
    float min_logLum,
    float factor,
    unsigned int* const d_bins,
    size_t num_bins) {
  const int tid = threadIdx.x;
  int idx = tid;

  extern __shared__ unsigned int bins[];
  while (idx < num_bins) {
    bins[idx] = 0;
    idx += blockDim.x;
  }
  __syncthreads();
#pragma unroll
  for (int i = 0, idx = tid; i < reads_per_thread && idx < total_num; i++) {
    float val = d_input[idx];
    unsigned int bin =
        min(static_cast<unsigned int>(num_bins - 1),
            static_cast<unsigned int>((val - min_logLum) * factor));
    atomicAdd(&bins[bin], 1);
    idx += blockDim.x;
  }
  __syncthreads();
  idx = tid;
  while (idx < num_bins) {
    d_bins[idx] = bins[idx];
    idx += blockDim.x;
  }
}

#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
__global__ void cdf_block(
    unsigned int* d_histo,
    unsigned int* d_cdf,
    size_t N,
    unsigned int* sums) {
  extern __shared__ unsigned int temp[];
  int tid = threadIdx.x;
  int leaf_num = blockDim.x * 2;
  int block_offset = blockIdx.x * leaf_num;
  int ai = tid;
  int bi = tid + blockDim.x;
  int bank_offset_ai = CONFLICT_FREE_OFFSET(ai);
  int bank_offset_bi = CONFLICT_FREE_OFFSET(bi);
  temp[ai + bank_offset_ai] =
      ai + block_offset < N ? d_histo[ai + block_offset] : 0;
  temp[bi + bank_offset_bi] =
      bi + block_offset < N ? d_histo[bi + block_offset] : 0;
  __syncthreads();

  int offset = 1;
  for (int d = leaf_num >> 1; d > 0; d >>= 1) {
    if (tid < d) {
      int ai = offset * (2 * tid + 1) - 1;
      int bi = offset * (2 * tid + 2) - 1;
      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);
      temp[bi] += temp[ai];
    }
    __syncthreads();
    offset *= 2;
  }
  if (tid == 0) {
    int last_idx = leaf_num - 1 + CONFLICT_FREE_OFFSET(leaf_num - 1);
    sums[blockIdx.x] = temp[last_idx];
    temp[last_idx] = 0;
  }
  __syncthreads();

  for (int d = 1; d < leaf_num; d *= 2) {
    offset >>= 1;
    if (tid < d) {
      int ai = offset * (2 * tid + 1) - 1;
      int bi = offset * (2 * tid + 2) - 1;
      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);
      int v = temp[ai];
      temp[ai] = temp[bi];
      temp[bi] += v;
    }
    __syncthreads();
  }
  if (ai + block_offset < N)
    d_cdf[ai + block_offset] = temp[ai + bank_offset_ai];
  if (bi + block_offset < N)
    d_cdf[bi + block_offset] = temp[bi + bank_offset_bi];
}

void your_histogram_and_prefixsum(
    const float* const d_logLuminance,
    unsigned int* const d_cdf,
    float& min_logLum,
    float& max_logLum,
    const size_t numRows,
    const size_t numCols,
    const size_t numBins) {
  // TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */
  size_t total_num = numRows * numCols;

  auto res = reduce_min_max<1024>(d_logLuminance, total_num);
  min_logLum = res.x;
  max_logLum = res.y;

  float range = max_logLum - min_logLum;
  float factor = numBins / range;
  unsigned int* d_bins;
  checkCudaErrors(cudaMalloc(&d_bins, sizeof(unsigned int) * numBins));

  int reads_per_thread = (total_num + 1023) / 1024;
  histogram<<<1, 1024, sizeof(unsigned int) * numBins>>>(
      d_logLuminance,
      total_num,
      reads_per_thread,
      min_logLum,
      factor,
      d_bins,
      numBins);
  checkCudaErrors(cudaGetLastError());
  unsigned int* h_bins = new unsigned int[numBins];
  checkCudaErrors(cudaMemcpy(
      h_bins, d_bins, sizeof(unsigned int) * numBins, cudaMemcpyDeviceToHost));

  unsigned int* d_sums;
  int block_num = numBins / 1024;
  int thread_num = 512;
  checkCudaErrors(cudaMalloc(&d_sums, sizeof(unsigned int) * block_num));
  cdf_block<<<1, 512, (1024 + 32) * sizeof(unsigned int)>>>(
      d_bins, d_cdf, numBins, d_sums);
  unsigned int* h_cdf = new unsigned int[numBins];
  checkCudaErrors(cudaMemcpy(
      h_cdf, d_cdf, sizeof(unsigned int) * numBins, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(d_bins));
}
