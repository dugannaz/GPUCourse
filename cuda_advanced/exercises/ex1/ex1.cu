#include <cuda.h>
#include <iostream>
#include <sys/time.h>
#include <stdio.h>

using namespace std;

/* Overlapping data transfers and kernel execution
 * - pinned memory
 * - streams
 * - different strategies depending on concurrent data transfers enabled or not
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


	// check if device can overlap data transfers with computation
	int deviceCount; 
	cudaGetDeviceCount(&deviceCount); 
	int device; 
	for (device = 0; device < deviceCount; ++device) { 
		cudaDeviceProp deviceProp; 
		cudaGetDeviceProperties(&deviceProp, device); 
		printf("Device %d has compute capability %d.%d.\n", device, 
			deviceProp.major, deviceProp.minor);
		cout << " asyncEngineCount = " << deviceProp.deviceOverlap << endl;
	}
	/************************/

        // side length of square matrix (in slide: N)
        int side = 2048;

        // number of elements in a single matrx
        int n = side*side;

        // number of matrices to transpose (in slide: M)
        int nTranspose = 96;

        // number of transpose operations on a single matrix
        int nreps = 20;


	// (*) Try with different nStream (nTranspose = integerConstant * nStream) 
	int nStream = 4;

	// (*) define streams here
      

	// create events
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

	// (*) modify here to allocate pinned host memory with cudaMallocHost 
	double *data = (double*)malloc(nTranspose * n * sizeof(double));
     
	// data initialization
	for (int j=0; j<nTranspose; j++)	   
	for (int i=0; i<n; i++) {
                data[i+j*n] = double(i+j*n);
        }

	double *data_dev;
	// (*) modify device memory allocation size according to nStream
        cudaMalloc((void**) &data_dev, 2 * n * sizeof(double));

	dim3 grid(side/16,side/16,1);
        dim3 threads(16,16,1);

	// record start event
        cudaEventRecord(start, 0);

	for (int i=0; i<nTranspose; i++) {

		// (*) modify following lines for overlaping computation and data transfers
		//     (Since nStream is less than nTranspose you will need an alternating
	   	//	device pointer offset)	
		cudaMemcpy(data_dev, data + i*n, n * sizeof(double), cudaMemcpyHostToDevice);

		transposeNaive <<< grid, threads >>>(data_dev + n, data_dev, side, side, nreps);

		cudaMemcpy(data + i*n, data_dev + n, n * sizeof(double), cudaMemcpyDeviceToHost);
	}

	// record stop event
        cudaEventRecord(stop, 0);

        // elapsed time
        cudaEventSynchronize(stop);
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, start, stop);
        cout << "Comp time = " << elapsedTime/1000.0 << endl;      

	// destroy events
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

	// (*) Destroy streams here


	cudaFree(data_dev);

	cout << "value check = " << data[n+5467] << endl;
	
	// (*) modify here to free pinned host memory
	free(data);
}

