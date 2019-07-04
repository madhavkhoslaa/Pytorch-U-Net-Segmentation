/* fragment gpu RAM by allocating a bunch of blocks and then releasing some in between, creating holes
   then try to allocate more than the size of the largest hole, but less than total free memory
   it appears that CUDA succeeds
   conclusiong: cudaMalloc it's not allocating contiguous memory
*/
#include <stdio.h>
#include <unistd.h>
#include <cuda.h>

const size_t Mb = 1<<20; // Assuming a 1Mb page size here

#define DSIZE0  410000000ULL //  ~400MB
#define DSIZE1 3144000000ULL // ~3000MB
#define DSIZE2  524000000ULL //  ~500MB
#define DSIZE3  630000000ULL //  ~600MB

void can_allocate() {
  size_t total;
  size_t avail;
  cudaError_t cuda_status = cudaMemGetInfo(&avail, &total);
  if ( cudaSuccess != cuda_status ) {
    printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );
    exit(EXIT_FAILURE);
  }

  printf("free: %.f, total %.f\n", (double)avail/Mb, (double)total/Mb);

  int *buf_d = 0;
  size_t nwords = total / sizeof(int);
  size_t words_per_Mb = Mb / sizeof(int);

  /* the only way to measure how much memory is allocatable is by trial and
     error, cudaMemGetInfo's available memory information is not reliable */
  while (cudaMalloc((void**)&buf_d,  nwords * sizeof(int)) == cudaErrorMemoryAllocation) {
    cudaFree(buf_d);
    nwords -= words_per_Mb;
    if (nwords < words_per_Mb) {
      // signal no free memory
      break;
    }
  }
  cudaFree(buf_d);
  /* clear last error */
  printf("err2:  %d\n", (int)cudaGetLastError());

  printf("can allocate:  %.fMB\n", (double)nwords/words_per_Mb);
}

int main() {
    int *d0, *d1, *d2, *d3, *d4;

    //cudaSetDevice(0);

    /* starting with 8GB free */
    /* legend: [allocated]{free} */

    // init - prealloc 500MB, including ~100MB CUDA ctx
    // [0.5]{7.5} - free total=7.5
    cudaMalloc(&d0, DSIZE0);
    printf("err1:  %d\n", (int)cudaGetLastError());

    // [0.5][0.5]{7.0} - free total=7.0
    cudaMalloc(&d1, DSIZE2);
    printf("err1:  %d\n", (int)cudaGetLastError());

    // [0.5][0.5][3]{4.0} - free total=4.0
    cudaMalloc(&d2, DSIZE1);
    printf("err2:  %d\n", (int)cudaGetLastError());

    // [0.5][0.5][3][0.5]{3.5} - free total=3.5
    cudaMalloc(&d3, DSIZE2);
    printf("err3:  %d\n", (int)cudaGetLastError());

    // [0.5][0.5][3][0.5][3]{0.5} - free total=0.5
    cudaMalloc(&d4, DSIZE1);
    printf("err2:  %d\n", (int)cudaGetLastError());

    // [0.5]{0.5}[3][0.5][3]{0.5} - free total=1.0
    cudaFree(d1);
    printf("err4:  %d\n", (int)cudaGetLastError());

    // [0.5]{0.5}[3]{0.5}[3]{0.5} - free total=1.5
    cudaFree(d3);
    printf("err4:  %d\n", (int)cudaGetLastError());

    // here we should have 1.5GB free in total, with 3 fragments of 0.5GB
    // this should say 0.5GB, but it says 1.6GB - so it allocates over fragments
    can_allocate();

    // another way to check is we shouldn't be able to allocate say 1GB of contiguous memory
    cudaMalloc(&d1, 2*DSIZE2);
    printf("err2:  %d\n", (int)cudaGetLastError());

    // sanity check 2GB at 1.5G free should fail
    // this fails, good
    cudaMalloc(&d1, 4*DSIZE2);
    printf("err2:  %d\n", (int)cudaGetLastError());

    sleep(1000);  /* keep consuming RAM */

    return 0;
}


