#include <cuda.h>
#include <iostream>

using namespace std;

/* example for device function usage
 */
__device__ void addOne_block(double *blockData) {

	// thread id is enough for computation
	int t = threadIdx.x;
	
	blockData[t]++;
}

__global__ void addOne(int n, double *data) {

	int b = blockIdx.x;

	// each block gets its data pointer as function argument 
	addOne_block(data + b*blockDim.x);
}

int main() {

	int n = 2048;

	double *data = (double*) malloc(n * sizeof(double));
        for (int i=0; i<n; i++) {
                data[i] = (double)i;
        }

	double *data_dev;
        cudaMalloc((void**) &data_dev, n * sizeof(double));

	cudaMemcpy(data_dev, data, n * sizeof(double) , cudaMemcpyHostToDevice);

        dim3 nBlocks(32,1);
	dim3 nThreads(64,1,1);
        addOne <<< nBlocks, nThreads >>> (n,data_dev);
	
        cudaMemcpy(data, data_dev, n * sizeof(double) , cudaMemcpyDeviceToHost);

	cudaFree(data_dev);

	cout << "data[n-1] = " << data[n-1] << endl;
	free(data);
}



