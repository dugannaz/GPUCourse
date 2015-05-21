#include <cuda.h>
#include <iostream>
#include <sys/time.h>
#include <stdio.h>

using namespace std;

/* Concurrent kernel execution
 * - compare concurrent execution performance with serial execution
 * - effect of (number of blocks) and (number of multiprocessors)
 */

#define TILE_DIM    16
#define BLOCK_ROWS  16

__global__ void transposeNaive(double *odata, double* idata, int width, int height, int nreps)
{
  int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
  int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;

  int index_in  = xIndex + width * yIndex;
  int index_out = yIndex + height * xIndex;
  for (int r=0; r < nreps; r++) {
    for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
      odata[index_out+i] = idata[index_in+i*width];
    }
  }
}

int main() {


	// check if device support concurrent executions
	int deviceCount; 
	cudaGetDeviceCount(&deviceCount); 
	int device; 
	for (device = 0; device < deviceCount; ++device) { 
		cudaDeviceProp deviceProp; 
		cudaGetDeviceProperties(&deviceProp, device); 
		printf("Device %d has compute capability %d.%d.\n", device, 
			deviceProp.major, deviceProp.minor);
		cout << " concurrent kernel execution = " << deviceProp.concurrentKernels << endl;
	}
	/************************/

	// (*) Repeat for side = 32, 64, 2048
	//     (for side = 2048, set nTranspose = 8)	
	int side = 32;
	int n = side*side;

	int nTranspose = 96;

	int nreps = 200;

	int nStream = nTranspose;
	// define streams
      	cudaStream_t stream[nStream];
        for (int i=0; i<nStream; i++)
                cudaStreamCreate(&stream[i]);


	time_t sTime = time(NULL);
        struct timeval tt1, tt2;
        int ms;
        double fms;

	// allocate pinned host memory 
	double *data;
	cudaMallocHost((void**) &data, nTranspose * n * sizeof(double));
     
	// data initialization
	for (int j=0; j<nTranspose; j++)	   
	for (int i=0; i<n; i++) {
                data[i+j*n] = double(i+j*n);
        }

	double *data_dev;
	// device memory allocation
        cudaMalloc((void**) &data_dev, nStream * 2 * n * sizeof(double));

	dim3 grid(side/16,side/16,1);
        dim3 threads(16,16,1);

	// send data to device
        for (int i=0; i<nStream; i++) {
        
		int offset = i * n;

		cudaMemcpy(data_dev + offset*2, data + offset, n * sizeof(double), 
					cudaMemcpyHostToDevice);
                
	}

	cudaThreadSynchronize();
        gettimeofday( &tt1, NULL );

	// kernel executions : 
	// (*) for concurrent execution : stream[0] --> stream[i]
        for (int i=0; i<nStream; i++) {

		int offset = i * n;

        	transposeNaive <<< grid, threads, 0, stream[0]  >>>
				(data_dev + offset*2+n, data_dev + offset*2, side, side, nreps);
	}

	cudaThreadSynchronize();
        gettimeofday( &tt2, NULL );
	
	// get data back from device
        for (int i=0; i<nStream; i++) {

		int offset = i * n;

        	cudaMemcpy(data + offset, data_dev + offset*2 + n, n * sizeof(double), 
					cudaMemcpyDeviceToHost);
        }

	// timing
	ms = (tt2.tv_sec - tt1.tv_sec);
        ms = ms * 1000000 + (tt2.tv_usec - tt1.tv_usec);
        fms = ((double)ms)/1000000.0;
        cout << "Comp time = " << fms << endl;        

	// dDestroy streams 
	for (int i=0; i<nStream; i++)
                cudaStreamDestroy(stream[i]);


	cudaFree(data_dev);

	cout << "value check = " << data[n+5467] << endl;
	
	// free pinned host memory
	cudaFreeHost(data);
}

