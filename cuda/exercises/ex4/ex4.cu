#include <cuda.h>
#include <iostream>

#define nPerThread 16

using namespace std;

/* Synchronization
 * - Synchronize threads in a block
 */

__global__ void myKernel(int n, double *data) {

	int t = threadIdx.x;
	int nt = blockDim.x;	

	// initialize values
	for (int i=0; i<nPerThread; i++)
		data[nt*i+t] = double(nt*i+t);

	// (*) synchronize threads here


	// increment values with inverse order
	for (int i=0; i<nPerThread; i++)
		data[n-(nt*i+t)-1] += 1.0;
		
}

int main() {

	int nBlocks = 1;
	int nThreads = 512;

	int n = nThreads * nPerThread;

	double *data = (double*) malloc(n * sizeof(double));
        for (int i=0; i<n; i++) {
                data[i] = 0;
        }

	double *data_dev;
        cudaMalloc((void**) &data_dev, n * sizeof(double));

	cudaMemcpy(data_dev, data, n * sizeof(double) , cudaMemcpyHostToDevice);

	myKernel <<< nBlocks, nThreads  >>>(n, data_dev);

        cudaMemcpy(data, data_dev, n * sizeof(double) , cudaMemcpyDeviceToHost);

	cudaFree(data_dev);

	cout << "data[n-1] = " << data[n-1] << endl;
	free(data);
}



