#include <cuda.h>
#include <iostream>

using namespace std;

__global__ void addOne(double *a) {

	int b = blockIdx.x;
	int t = threadIdx.x;
	
	int i = b * blockDim.x + t;

	a[i]++;
}

/* cuda interface function for fortran
 * (note that function name should have an additonal "_" at the end)
 */
extern "C" void kernel_wrapper_(int *n_p, int *nb, int *nt) {

	int n = *n_p;

	double *data = (double*) malloc(n * sizeof(double));
        for (int i=0; i<n; i++) {
                data[i] = (double)i;
        }

	double *data_dev;
        cudaMalloc((void**) &data_dev, n * sizeof(double));

	cudaMemcpy(data_dev, data, n * sizeof(double) , cudaMemcpyHostToDevice);

        dim3 nBlocks(*nb,1,1);
	dim3 nThreads(*nt,1,1);
        addOne <<< nBlocks, nThreads >>> (data_dev);

        cudaMemcpy(data, data_dev, n * sizeof(double) , cudaMemcpyDeviceToHost);

	cudaFree(data_dev);

	cout << "data[n-1] = " << data[n-1] << endl;
	free(data);
}



