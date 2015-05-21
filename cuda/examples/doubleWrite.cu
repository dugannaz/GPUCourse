#include <cuda.h>
#include <iostream>
#include <sys/time.h>

using namespace std;

/* example showing multiple threads writing same memory location
 */
__global__ void doubleWrite(int n, double *a) {

	int t = threadIdx.x;

	// access order is not defined. result would change in different runs
	a[0] = t;
}

int main() {

	int n = 128*128*128*4;

	double *data = (double*) malloc(n * sizeof(double));
        for (int i=0; i<n; i++) {
                data[i] = i;
        }

	double *data_dev;
        cudaMalloc((void**) &data_dev, n * sizeof(double));

        int nBlocks = 1;
	int nThreads = 512;

	// runs the kernel 100 times
	for (int i=0; i<100; i++) {

		cudaMemcpy(data_dev, data, n * sizeof(double) , cudaMemcpyHostToDevice);

		doubleWrite <<< nBlocks, nThreads  >>>(n, data_dev);

		cudaMemcpy(data, data_dev, n * sizeof(double) , cudaMemcpyDeviceToHost);

		cout << "data[0] = " << data[0] << endl;
	}
	cudaFree(data_dev);
	free(data);
}



