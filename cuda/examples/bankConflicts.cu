#include <cuda.h>
#include <iostream>
#include <sys/time.h>
#include <stdio.h>

using namespace std;

/* Bank conflict example
 */

// (*) also compare with: TILE_DIM = BLOCK_ROWS = 32
#define TILE_DIM    16
#define BLOCK_ROWS  16

__global__ void transposeCoalesced(double *odata, double *idata, int width, int height, int nreps)
{

  // (*) Uncomment +1 to avoid bank conflicts
  __shared__ float tile[TILE_DIM][TILE_DIM /* +1 */];

  int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
  int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
  int index_in = xIndex + (yIndex)*width;

  xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
  yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
  int index_out = xIndex + (yIndex)*height;

  for (int r=0; r < nreps; r++) {
    for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
      tile[threadIdx.y+i][threadIdx.x] = idata[index_in+i*width];
    }

    __syncthreads();

    for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
      odata[index_out+i*height] = tile[threadIdx.x][threadIdx.y+i];
    }
  }
}


int main() {

	// number of kernel calls
	int nTranspose = 1;

	// number of transposes inside kernel after copying to shared memory.
	int nreps = 20;

	time_t sTime = time(NULL);
        struct timeval tt1, tt2;
        int ms;
        double fms;

	int side = 2048;
	int n = side*side;

	double *data = (double*)malloc(2*n * sizeof(double));
     	   
	for (int i=0; i<n; i++) {
                data[i] = double(i);
		data[i+n] = 0;
        }

	double *data_dev;
        cudaMalloc((void**) &data_dev, 2 * n * sizeof(double));

	dim3 grid(side/TILE_DIM,side/TILE_DIM,1);
        dim3 threads(TILE_DIM,BLOCK_ROWS,1);

	cudaMemcpy(data_dev, data, 2*n * sizeof(double), cudaMemcpyHostToDevice);

	cudaThreadSynchronize();
	gettimeofday( &tt1, NULL );
		
	 for (int i=0; i<nTranspose; i++)
		transposeCoalesced <<< grid, threads >>>(data_dev+n, data_dev, side, side, nreps);

	cudaThreadSynchronize();
	gettimeofday( &tt2, NULL );	

	cudaMemcpy(data, data_dev + n, n * sizeof(double), cudaMemcpyDeviceToHost);
	

	ms = (tt2.tv_sec - tt1.tv_sec);
        ms = ms * 1000000 + (tt2.tv_usec - tt1.tv_usec);
        fms = ((double)ms)/1000000.0;
        cout << "Comp time = " << fms << endl;        

	cudaFree(data_dev);

	cout << "data[145] = " << data[145] << endl;
	
	free(data);
}

