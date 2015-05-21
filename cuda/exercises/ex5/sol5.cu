#include <cuda.h>
#include <iostream>
#include <sys/time.h>

using namespace std;

/* Simple Cuda Program: 2D block version
 * - map 1D thread block to 2D data
 * - use 2D thread block
 * - effect of non-optimal block size 
 */

// (*3*) set dataX to 17
#define dataX 16

#define nThreadsX (dataX*dataX)

#define BLOCK_DATA(i,j) block_data[(i)*dataX+(j)]

__global__ void addOne(double *data) {

	int b = blockIdx.x;

	// pointer to block data 
	double *block_data = data + b*nThreadsX;

	// (*1*) Interchange the definitions of tx and ty 
	// (*2*) use threadIdx.x and threadIdx.y (for original coalesced access version)
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	// access data as 2D
	for (int i=0;i<100000; i++)
		BLOCK_DATA(ty,tx)++;
}

int main() {

	// time variables
	time_t sTime = time(NULL);
        struct timeval tt1, tt2;
        int ms;
        double fms;

	// (*3*) set data size to 4624 (17*17 * 16) 
	int n = 4096;

	double *data = (double*) malloc(n * sizeof(double));
        for (int i=0; i<n; i++) {
                data[i] = (double)i;
        }

	double *data_dev;
        cudaMalloc((void**) &data_dev, n * sizeof(double));

	cudaMemcpy(data_dev, data, n * sizeof(double) , cudaMemcpyHostToDevice);

        dim3 nBlocks(n/(nThreadsX),1);

	// (*2*) modify here to make a 2D block (dataX x dataX)
	dim3 nThreads(dataX,dataX,1);

	gettimeofday( &tt1, NULL );	

        addOne <<< nBlocks, nThreads >>> (data_dev);
	
	cudaThreadSynchronize();
        gettimeofday( &tt2, NULL );

        cudaMemcpy(data, data_dev, n * sizeof(double) , cudaMemcpyDeviceToHost);

	cudaFree(data_dev);

	// time calculation
	ms = (tt2.tv_sec - tt1.tv_sec);
        ms = ms * 1000000 + (tt2.tv_usec - tt1.tv_usec);
        fms = ((double)ms)/1000000.0;
        cout << "kernel run time = " << fms << endl;

	cout << "data[n-1] = " << data[n-1] << endl;
	free(data);
}



