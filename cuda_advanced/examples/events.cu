#include <cuda.h>
#include <iostream>
#include <sys/time.h>

using namespace std;

/* Timing with CUDA events
 */

__global__ void addOne(int n, double *data) {

	int nb = gridDim.x;
	int nt = blockDim.x;

	int compPerThread = n / (nb*nt);

	int b = blockIdx.x;
	int t = threadIdx.x;
	
	int i = b * blockDim.x + t;

	for (int j=0; j<compPerThread; j++)
		data[i+j*nb*nt]++;
}

int main() {

	// create evets
	cudaEvent_t start, stop; 
	cudaEventCreate(&start); 
	cudaEventCreate(&stop);

	time_t sTime = time(NULL);
        struct timeval tt1, tt2;
        int ms;
        double fms;

	int n = 128*128*128*4;

	double *data = (double*) malloc(n * sizeof(double));
        for (int i=0; i<n; i++) {
                data[i] = i;
        }

	double *data_dev;
        cudaMalloc((void**) &data_dev, n * sizeof(double));

	cudaMemcpy(data_dev, data, n * sizeof(double) , cudaMemcpyHostToDevice);
	
        int nBlocks = 32*32;
	int nThreads = 32;

	// record event (synchronization is not needed)
	cudaEventRecord(start, 0);
        
	addOne <<< nBlocks, nThreads  >>>(n, data_dev);

	// record event (synchronization is not needed)	
	cudaEventRecord(stop, 0);

	// make sure that stop event is recorded
	cudaEventSynchronize(stop);
 
	float elapsedTime; 
	// calculate elapsed time in miliseconds
	cudaEventElapsedTime(&elapsedTime, start, stop);

	cout << "Comp time (event) = " << elapsedTime/1000.0 << endl;

	// usual timing
	cudaThreadSynchronize();
        gettimeofday( &tt1, NULL );

	addOne <<< nBlocks, nThreads  >>>(n, data_dev);

	cudaThreadSynchronize();
	gettimeofday( &tt2, NULL );

	ms = (tt2.tv_sec - tt1.tv_sec);
        ms = ms * 1000000 + (tt2.tv_usec - tt1.tv_usec);
        fms = ((double)ms)/1000000.0;
        cout << "Comp time (gettimeofday) = " << fms << endl;

        cudaMemcpy(data, data_dev, n * sizeof(double) , cudaMemcpyDeviceToHost);

	cudaEventDestroy(start); 
	cudaEventDestroy(stop);

	cudaFree(data_dev);

	cout << "data[n-1] = " << data[n-1] << endl;
	free(data);
}



