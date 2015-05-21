#include <cuda.h>
#include <iostream>

using namespace std;

/* 2D thread block version of addOne kernel
 */
__global__ void addOne(double *data) {

	int b = blockIdx.x;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	
	// 2D threads are mapped to 1D memory
	int i = b * (blockDim.x * blockDim.y) + (ty * blockDim.x + tx);

	data[i]++;
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
	dim3 nThreads(16,16,1);
        addOne <<< nBlocks, nThreads >>> (data_dev);
	
        cudaMemcpy(data, data_dev, n * sizeof(double) , cudaMemcpyDeviceToHost);

	cudaFree(data_dev);

	cout << "data[n-1] = " << data[n-1] << endl;
	free(data);
}



