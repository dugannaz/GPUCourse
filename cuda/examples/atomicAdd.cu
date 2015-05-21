#include <cuda.h>
#include <iostream>
#include <sys/time.h>

using namespace std;

/* example for atomic function usage
 */

__global__ void atomic(int n, float *a) {

	//a[0] += 1.0f; // gives wrong result
	
	// instead use atomic function
	atomicAdd(&a[0], 1.0f); 
}

int main() {

	int n = 1024;

	float *data = (float*) malloc(n * sizeof(float));
        for (int i=0; i<n; i++) {
                data[i] = (float)i;
        }

	float *data_dev;
        cudaMalloc((void**) &data_dev, n * sizeof(float));

	cudaMemcpy(data_dev, data, n * sizeof(float) , cudaMemcpyHostToDevice);
	cudaError_t error = cudaGetLastError();
	cout << "copy to device = " << error << " : " << cudaGetErrorString(error) << endl;

        int nBlocks = 1;
	int nThreads = 1024;

	atomic <<< nBlocks, nThreads  >>>(n, data_dev);

	error = cudaGetLastError();
        cout << "run kernel = " << error << " : " << cudaGetErrorString(error) << endl;

        cudaMemcpy(data, data_dev, n * sizeof(float) , cudaMemcpyDeviceToHost);
	error = cudaGetLastError();
        cout << "copy from device = " << error << " : " << cudaGetErrorString(error) << endl;

	cudaFree(data_dev);

	cout << "data[0] = " << data[0] << endl;
	free(data);
}



