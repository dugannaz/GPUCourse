#include <cuda.h>
#include <iostream>
#include <sys/time.h>

using namespace std;

/* Simple Cuda program for arbitrary array size
 * - block, thread variables
 * - data transfer and kernel run times
 * - kernel execution parameters: effect on performance  
 */

// (*) you should pass the length of array (n) to this function
__global__ void addOne(int n, double *data) {

	// (*) modify the following lines to work on whole array when (n) is not 
        //     equal to (blockDim.x * gridDim.x) but a multiple of it.
	
	int nb = gridDim.x;
	int nt = blockDim.x;

	int compPerThread = n / (nb*nt);

	int b = blockIdx.x;
	int t = threadIdx.x;
	
	int i = (b * nt + t)*compPerThread;

	for (int j=0; j<compPerThread; j++)
		data[i+j]++;
}

// cpu addOne function
void addOne_cpu(int n, double *data) {

	for (int i=0; i<n; i++)
		data[i]++;
}

int main() {

	struct timeval t1, t2, t3, t4, t5;
        float dt;

	int n = 8388608;

	double *data = (double*) malloc(n * sizeof(double));
        for (int i=0; i<n; i++) {
                data[i] = 0;
        }

	// (*) have timing with different configuration parameters
        dim3 nBlocks(8192,1,1);
	dim3 nThreads(1024,1,1);

	double *data_dev;
        cudaMalloc((void**) &data_dev, n * sizeof(double));

	//(*) if necessary synchronize threads before getting system time
        gettimeofday(&t1, NULL);

	cudaMemcpy(data_dev, data, n * sizeof(double) , cudaMemcpyHostToDevice);
	
        gettimeofday(&t2, NULL);

	// (*) call kernel
        addOne <<< nBlocks, nThreads  >>>(n, data_dev);

	cudaThreadSynchronize();
        gettimeofday(&t3, NULL);

        cudaMemcpy(data, data_dev, n * sizeof(double) , cudaMemcpyDeviceToHost);

        gettimeofday(&t4, NULL);

	// cpu computation
	addOne_cpu(n, data);

	gettimeofday(&t5, NULL);

	cudaFree(data_dev);

	cout << "data[n-1] = " << data[n-1] << endl;
	free(data);

	// timing
        dt = (t2.tv_sec - t1.tv_sec + 1.0e-6 * (t2.tv_usec - t1.tv_usec));
	cout << "host -> device transfer time = " << dt << endl;

	dt = (t3.tv_sec - t2.tv_sec + 1.0e-6 * (t3.tv_usec - t2.tv_usec));
        cout << "kernel run time = " << dt << endl;

	dt = (t4.tv_sec - t3.tv_sec + 1.0e-6 * (t4.tv_usec - t3.tv_usec));
        cout << "device -> host transfer time = " << dt << endl;

	dt = (t5.tv_sec - t4.tv_sec + 1.0e-6 * (t5.tv_usec - t4.tv_usec));
        cout << "cpu comp time = " << dt << endl;

}



