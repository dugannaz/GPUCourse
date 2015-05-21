#include <cuda.h>
#include <iostream>

using namespace std;

/* Simple Cuda Program
 * - Compile and run 
 * - kernel execution parameters
 * - error check 
 */
__global__ void addOne(double *a) {

	int b = blockIdx.x;
	int t = threadIdx.x;
	
	int i = b * blockDim.x + t;

	a[i]++;
}

int main() {

	int n = 2048;

	double *data = (double*) malloc(n * sizeof(double));
        for (int i=0; i<n; i++) {
                data[i] = (double)i;
        }

	/**** CUDA ****/
	double *data_dev;
        cudaMalloc((void**) &data_dev, n * sizeof(double));

	cudaMemcpy(data_dev, data, n * sizeof(double) , cudaMemcpyHostToDevice);

	// (*) Change grid size (nBlocks) and block size (nThreads)
        dim3 nBlocks(32,1,1);
	dim3 nThreads(64,1,1);
        addOne <<< nBlocks, nThreads >>> (data_dev);
	
	// (*) Uncomment below lines to check kernel run status
	//cudaError_t error = cudaGetLastError();
	//cout << "error code = " << error << " : " << cudaGetErrorString(error) << endl;

        cudaMemcpy(data, data_dev, n * sizeof(double) , cudaMemcpyDeviceToHost);

	cudaFree(data_dev);
	/**** CUDA ****/

	cout << "data[n-1] = " << data[n-1] << endl;
	free(data);
}



