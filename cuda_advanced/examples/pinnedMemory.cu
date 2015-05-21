#include <cuda.h>
#include <iostream>
#include <sys/time.h>

using namespace std;

/* Pinned (page locked) memory example
 */
__global__ void addOne(int n, double *data) {

	int nb = gridDim.x;
	int nt = blockDim.x;

	int compPerThread = n / (nb*nt);

	int b = blockIdx.x;
	int t = threadIdx.x;
	
	int i = (b * nt + t)*compPerThread;

	for (int j=0; j<compPerThread; j++)
		data[i+j]++;
}

int main() {

	time_t sTime = time(NULL);
        struct timeval tt1, tt2;
        int ms;
        double fms;

	int n = 8388608;

	// paged memory is allocated
	double *data;
	data = (double*) malloc(n * sizeof(double));

	// pinned (page locked) memory allocated
	double *data1;
	cudaMallocHost((void**) &data1, n * sizeof(double));

        for (int i=0; i<n; i++) {
                data[i] = 0.0;
        }

	for (int i=0; i<n; i++) {
                data1[i] = 0.0;
        }

	double *data_dev;
        cudaMalloc((void**) &data_dev, n * sizeof(double));

	dim3 nBlocks(1024,1,1);
	dim3 nThreads(256,1,1);

	// paged memory :
	// timing for data transfer, kernel execution, data transfer back
	cudaThreadSynchronize();
        gettimeofday( &tt1, NULL );
        
	cudaMemcpy(data_dev, data, n * sizeof(double) , cudaMemcpyHostToDevice);

	addOne <<< nBlocks, nThreads  >>>(n, data_dev);

        cudaMemcpy(data, data_dev, n * sizeof(double) , cudaMemcpyDeviceToHost);

	cudaThreadSynchronize();
        gettimeofday( &tt2, NULL );

	ms = (tt2.tv_sec - tt1.tv_sec);
        ms = ms * 1000000 + (tt2.tv_usec - tt1.tv_usec);
        fms = ((double)ms)/1000000.0;
        cout << "Paged memory : elapsed Time = " << fms << endl;

	cout << "data[n-1] = " << data[n-1] << endl;

	// pinned memory :
	// timing for data transfer, kernel execution, data transfer back
	cudaThreadSynchronize();
        gettimeofday( &tt1, NULL );
        
	cudaMemcpy(data_dev, data1, n * sizeof(double) , cudaMemcpyHostToDevice);

	addOne <<< nBlocks, nThreads  >>>(n, data_dev);

        cudaMemcpy(data1, data_dev, n * sizeof(double) , cudaMemcpyDeviceToHost);

	cudaThreadSynchronize();
        gettimeofday( &tt2, NULL );

	ms = (tt2.tv_sec - tt1.tv_sec);
        ms = ms * 1000000 + (tt2.tv_usec - tt1.tv_usec);
        fms = ((double)ms)/1000000.0;
        cout << "Pinned memory : elapsed Time = " << fms << endl;

	cout << "data1[n-1] = " << data1[n-1] << endl;

	cudaFree(data_dev);

	// paged memory is freed
	free(data);
	// pinned memory is freed
	cudaFreeHost(data1);
}



