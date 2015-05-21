#include <cuda.h>
#include <iostream>
#include <sys/time.h>

using namespace std;

// double precision atomic add function
// (there is no intrinsic double precision atomicAdd)
__device__ double atomicAdd_d(double* address, double val) { 

	unsigned long long int* address_as_ull = (unsigned long long int*)address; 
	unsigned long long int old = *address_as_ull, assumed; 
	do { 
		assumed = old; 
		old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val 
						+ __longlong_as_double(assumed))); 
	} while (assumed != old); 

	return __longlong_as_double(old);
}


__global__ void atomic(int n, double *a) {

	//a[0] += 1.0; // gives wrong result
	
	// instead use atomic function
	atomicAdd_d(&a[0], 1.0); 
}

int main() {

	int n = 1024;

	double *data = (double*) malloc(n * sizeof(double));
        for (int i=0; i<n; i++) {
                data[i] = (double)i;
        }

	double *data_dev;
        cudaMalloc((void**) &data_dev, n * sizeof(double));

	cudaMemcpy(data_dev, data, n * sizeof(double) , cudaMemcpyHostToDevice);
	cudaError_t error = cudaGetLastError();
	cout << "copy to device = " << error << " : " << cudaGetErrorString(error) << endl;

        int nBlocks = 1;
	int nThreads = 1024;

	atomic <<< nBlocks, nThreads  >>>(n, data_dev);

	error = cudaGetLastError();
        cout << "run kernel = " << error << " : " << cudaGetErrorString(error) << endl;

        cudaMemcpy(data, data_dev, n * sizeof(double) , cudaMemcpyDeviceToHost);
	error = cudaGetLastError();
        cout << "copy from device = " << error << " : " << cudaGetErrorString(error) << endl;

	cudaFree(data_dev);

	cout << "data[0] = " << data[0] << endl;
	free(data);
}



