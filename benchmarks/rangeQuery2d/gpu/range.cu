#include <algorithm>
#include <limits>
#include <iostream>
#include <cfloat>
#include "parlay/primitives.h"
#include "parlay/internal/get_time.h"
#include "parlay/parallel.h"
#include "parlay/io.h"
#include "common/time_loop.h"
#include "common/IO.h"
#include "common/geometryIO.h"
#include "common/parse_command_line.h"

#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/merge.h>

#define BLOCK_SIZE 1024

using namespace std;

#define CUDA_CHECK(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      cerr << "Failed to run stmt " << #stmt << "\n" << cudaGetErroString(err) << "\n"; \
      return -1;                                                          \
    }                                                                     \
  } while (0)

using coord = double;
struct point {
  coord x, y;
  __host__ __device__ point(coord x, coord y) : x(x), y(y) {}
  __host__ __device__ point() {}
  __host__ __device__ bool operator < (const point& a) const {
    return x < a.x;
  }
};
struct query {coord x1, x2, y1, y2;};
using Points = parlay::sequence<point>;
using Queries = parlay::sequence<query>;

long range(Points const &points, Queries const &queries, bool verbose);

__device__ int lowerbound(double* begin, size_t size, double target) {
  int left = 0, right = size - 1;
  while (left < right) {
    int mid = (left + right) / 2;
    if (begin[mid] < target) left = mid + 1;
    else right = mid;
  }
  return left;
}


__global__ void canonicalInput(point* points, double* augs, int N, int tree_size){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N) {
    for (int rank = 0; rank < tree_size -1 ; rank++)
      augs[rank * N + tid] = points[tid].y;
  }
}

__device__ void partialMerge(double* A, size_t a_size, double* B, size_t b_size, double* C) {
  size_t ia = 0, ib = 0, ic = 0;
  while (ia < a_size && ib < b_size) {
    if (A[ia] < B[ib]) {
      C[ic++] = A[ia++];
    } else {
      C[ic++] = B[ib++];
    }
  }
  if (ia == a_size) {
    while (ib < b_size) {
      C[ic++] = B[ib++];
    }
  } else {
    while (ia < a_size) {
      C[ic++] = A[ia++];
    }
  }
}

__global__ void canonicalMerge(double* augs, int N, int rank) {
  size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

  double* src = augs + rank * N;
  double* dst = src + N;
  if ((tid << (rank + 1)) < long(N)) {
    size_t size = 1 << rank;
    size_t ia = tid << (rank + 1);
    size_t ib = ia + size;
    size_t a_size = min(size, N - ia);
    size_t b_size = N > ib ? min(size, N - ib) : 0;
    partialMerge(src + ia, a_size, src + ib, b_size, dst + ia);
  }
}

__global__ void parallelMerge(double *augs, int N, int rank) {
  size_t size = 1 << rank;
  size_t tile = size / BLOCK_SIZE;
  double* src_offset = augs + rank * N;
  double* dst_offset = src_offset + N;
  size_t ia = blockIdx.x << (rank + 1);
  if (ia >= long(N)) return;
  size_t ib = ia + size;
  size_t a_size = min(size, N - ia);
  size_t b_size = N > ib ? min(size, N - ib) : 0;
  
  double* new_a = src_offset + ia;
  double* new_b = src_offset + ib;
  double* new_c = dst_offset + ia;

  if (b_size == 0) {
    for (size_t t = tile * threadIdx.x; t < min((tile + 1) * threadIdx.x, a_size); t++) {
      new_c[t] = new_a[t];
    }
  } else {
    double* tiled_ia = new_a + tile * threadIdx.x;
    size_t next_lb = threadIdx.x == BLOCK_SIZE - 1 ? b_size : lowerbound(new_b, b_size, *(tiled_ia + tile));
    size_t lb = lowerbound(new_b, b_size, *tiled_ia);
    double* tiled_ib = new_b + lb;
    size_t tiled_ib_size = next_lb - lb;
    partialMerge(tiled_ia, tile, new_b + lb, tiled_ib_size, new_c + tile * threadIdx.x + lb);
  }
}

__global__ void constructRangeTree(point* points, int N, int n_log, int tree_size, int* lefts, 
  int* rights, int* y_offsets, int* cnts) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int size = N, acc = 0, prev_acc = -1;
  for (int rank = 0; rank <= n_log; rank++) {
    if (tid + acc >= tree_size) return;
    if (tid < size) {
      y_offsets[acc + tid] = tid << rank;
      cnts[acc + tid] = min(1 << rank, N - (tid << rank));
      lefts[acc + tid] = prev_acc == -1 ? -1 : prev_acc + tid * 2;
      rights[acc + tid] = prev_acc == -1 ? -1 : min(acc - 1, prev_acc + tid * 2 + 1);
    }
    prev_acc = acc;
    acc += size;
    size = (size + 1) / 2;
  }
}

__device__ int get_query_count(double* aug, size_t begin, size_t size, double y1, double y2) {
  if (size == 1) {
    return (y1 <= aug[begin] && aug[begin] < y2) ? 1 : 0;
  } else {
    return lowerbound(aug + begin, size, y2) - lowerbound(aug + begin, size, y1);
  }
}

__global__ void reduction(const long *gArr, int arraySize, long *gOut) {
    int thIdx = threadIdx.x;
    int gthIdx = thIdx + blockIdx.x*BLOCK_SIZE;
    const int gridSize = BLOCK_SIZE*gridDim.x;
    long sum = 0;
    for (int i = gthIdx; i < arraySize; i += gridSize)
        sum += gArr[i];
    __shared__ long shArr[BLOCK_SIZE];
    shArr[thIdx] = sum;
    __syncthreads();
    for (int size = BLOCK_SIZE/2; size>0; size/=2) { //uniform
        if (thIdx<size)
            shArr[thIdx] += shArr[thIdx+size];
        __syncthreads();
    }
    if (thIdx == 0)
        gOut[blockIdx.x] = shArr[0];
}

__global__ void rangeQuery(point* points, int N, int n_log, int tree_size, query* queries, int Q,
    int* lefts, int* rights, int* y_offsets, double* augs, int* cnts, long* sums) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < Q) {
    double x1 = queries[tid].x1, x2 = queries[tid].x2, y1 = queries[tid].y1, y2 = queries[tid].y2;
    int x1_idx = tree_size - 1, x2_idx = x1_idx;
    long acc = 0;
    bool split_ancestor = false;
    for (int rank = n_log; rank >= 0; rank--) {
      int x1_left = lefts[x1_idx], x2_left = lefts[x2_idx];
      int x1_right = rights[x1_idx], x2_right = rights[x2_idx];
      if (rank == 0) {
        sums[tid] = acc + get_query_count(augs, x1_idx, 1, y1, y2) + get_query_count(augs, x2_idx, 1, y1, y2);
        break;
      }
      if (x1_left == -1 || x2_left == -1 || x1_right == -1 || x2_right == -1) {
        break;
      }
      bool left_branch = points[y_offsets[x1_left] + cnts[x1_left] - 1].x < x1;
      bool right_branch = points[y_offsets[x2_right]].x <= x2;
      if (split_ancestor) {
        if (!left_branch) 
          acc += get_query_count(augs, (rank - 1) * N + y_offsets[rights[x1_idx]], cnts[rights[x1_idx]], y1, y2);
        if (right_branch) 
          acc += get_query_count(augs, (rank - 1) * N + y_offsets[lefts[x2_idx]], cnts[lefts[x2_idx]], y1, y2);
      }
      x1_idx = left_branch ? x1_right : x1_left;
      x2_idx = right_branch ? x2_right : x2_left;
      split_ancestor |= left_branch != right_branch;
    }
  }
}


#define tcast(x) thrust::raw_pointer_cast(x.data())

long range(Points const &points, Queries const &queries, bool verbose) {
  parlay::internal::timer t("range", true);
  int numPoints = points.size();
  int numQueries = queries.size();
  int numPointsCeilLog = 32 - __builtin_clz(numPoints - 1);

  cudaError_t cudaStatus;

  int treeSize = 1;
  for (int i = numPoints; i > 1; i = (i + 1) / 2) {
    treeSize += i;
  }

  long result = 0;
  {
    query* deviceQueries;
    double* deviceAugs;
    long* devResult;
    int *deviceLefts, *deviceRights, *deviceYOffsets, *deviceCnts;
    long* deviceSums;
    t.next("ready");
  
    cudaMalloc((void**)&deviceQueries, sizeof(query) * numQueries); // 32n
    cudaMalloc((void**)&devResult, sizeof(long) * 64); 
    cudaMalloc((void**)&deviceAugs, numPoints * (1 + numPointsCeilLog) * sizeof(double)); // 8n * 27
    cudaMalloc((void**)&deviceLefts, sizeof(int) * treeSize); // 8n
    cudaMalloc((void**)&deviceRights, sizeof(int) * treeSize); // 8n
    cudaMalloc((void**)&deviceYOffsets, sizeof(int) * treeSize); // 8n
    cudaMalloc((void**)&deviceCnts, sizeof(int) * treeSize); // 8n
    cudaMalloc((void**)&deviceSums, sizeof(long) * numQueries); // 8n
    t.next("alloc");

    thrust::device_vector<point> devicePoints(points.begin(), points.end());
    cudaMemcpy(deviceQueries, queries.data(), sizeof(query) * numQueries, cudaMemcpyHostToDevice);
    t.next("H->D");

    thrust::sort(devicePoints.begin(), devicePoints.end());
    
    t.next("sort");

    int dimGrid = (numPoints + BLOCK_SIZE - 1) / BLOCK_SIZE;
    canonicalInput<<<dimGrid, BLOCK_SIZE>>>(tcast(devicePoints), deviceAugs, numPoints, numPointsCeilLog);
    cudaDeviceSynchronize();
    int elemCnt = (numPoints + 1) / 2;
    for (int rank = 0; rank < numPointsCeilLog - 1; rank++) {
      dimGrid = (elemCnt + BLOCK_SIZE - 1) / BLOCK_SIZE;
      if (rank < 12) {
        canonicalMerge<<<dimGrid, BLOCK_SIZE>>>(deviceAugs, numPoints, rank);
      } else {
        dimGrid = (numPoints + (1 << (rank + 1)) - 1) / (1 << (rank + 1));
        parallelMerge<<<dimGrid, BLOCK_SIZE>>>(deviceAugs, numPoints, rank);
      }
      elemCnt = (elemCnt + 1) / 2;
      cudaDeviceSynchronize();
      cudaStatus = cudaGetLastError();
      if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA error in merge: " << rank << " " << cudaGetErrorString(cudaStatus) << std::endl;
      }
    }
    t.next("merge");

    // find tree size
    dimGrid = (numPoints + BLOCK_SIZE - 1) / BLOCK_SIZE;
    constructRangeTree<<<dimGrid, BLOCK_SIZE>>>(tcast(devicePoints), numPoints, 
      numPointsCeilLog, treeSize, deviceLefts, deviceRights, deviceYOffsets, deviceCnts);
    cudaDeviceSynchronize();
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA error in build: " << cudaGetErrorString(cudaStatus) << std::endl;
    }
    t.next("tree");

    dimGrid = (numQueries + BLOCK_SIZE - 1) / BLOCK_SIZE;
    rangeQuery<<<dimGrid, BLOCK_SIZE>>>(tcast(devicePoints), numPoints, numPointsCeilLog, treeSize,
      deviceQueries, numQueries, deviceLefts, deviceRights, deviceYOffsets, deviceAugs, deviceCnts, deviceSums);
    cudaDeviceSynchronize();
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA error in query: " << cudaGetErrorString(cudaStatus) << std::endl;
    }
    t.next("query");
    reduction<<<64, BLOCK_SIZE>>>(deviceSums, numQueries, deviceSums);
    reduction<<<1, BLOCK_SIZE>>>(deviceSums, 64, deviceSums);
    cudaMemcpy(&result, deviceSums, sizeof(long), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    t.next("reduction & D->H");

    cudaFree(deviceQueries);
    cudaFree(devResult);
    cudaFree(deviceAugs);
    cudaFree(deviceLefts);
    cudaFree(deviceRights);
    cudaFree(deviceYOffsets);
    cudaFree(deviceCnts);
    cudaFree(deviceSums);
    t.next("clear");
  }
  cout << "total elapsed time : " << t.total_time() << endl;

  return result;
}


void timeRange(Points const &points, Queries const& queries,
	       int rounds, bool verbose, char* outFile) {
  cout << "start timeRange" << endl;
  long result;

  result = range(points, queries, verbose);

  cout << "total count = " << result << endl;
  if (outFile != NULL) parlay::chars_to_file(parlay::to_chars(result), outFile);
}

using pointx = point2d<coord>;

int main(int argc, char* argv[]) {
  commandLine P(argc,argv,"[-o <outFile>] [-r <rounds>] [-v] <inFile>");
  char* iFile = P.getArgument(0);
  char* oFile = P.getOptionValue("-o");
  bool verbose = P.getOption("-v");
  int rounds = P.getOptionIntValue("-r",0);

  parlay::sequence<pointx> A = readPointsFromFile<pointx>(iFile);
  size_t n = A.size();
  cout << n << endl;
  size_t num_q = n/3;
  auto points = parlay::map(A.cut(2 * num_q, rounds ? 2 * num_q + rounds : n), [&] (pointx pt) {return point{pt.x,pt.y};});
  auto queries = parlay::tabulate(num_q, [&] (size_t i) {			 
     coord x1 = A[2*i].x;
     coord y1 = A[2*i].y;
     coord x2 = A[2*i+1].x;
     coord y2 = A[2*i+1].y;
     query a{min(x1,x2), max(x1,x2), min(y1,y2), max(y1,y2)};
     return a;});
  timeRange(points, queries, rounds, verbose, oFile);
}


