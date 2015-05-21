#include <cuda.h>
#include <iostream>
#include <sys/time.h>

using namespace std;

// non coalesced 
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

// colaesced
__global__ void addOne1(int n, double *data) {

	int nb = gridDim.x;
	int nt = blockDim.x;

	int compPerThread = n / (nb*nt);

	int b = blockIdx.x;
	int t = threadIdx.x;
	
	int i = b * nt + t;

	for (int j=0; j<compPerThread; j++)
		data[i+j*nb*nt]++;
}

// coalesced (distribute data to blocks)
__global__ void addOne2(int n, double *data) {

	int nb = gridDim.x;
	int nt = blockDim.x;

	int compPerThread = n / (nb*nt);
	int blockDataSize = compPerThread*nt;

	int b = blockIdx.x;
	int t = threadIdx.x;	

	for (int j=0; j<compPerThread; j++)
		data[b*blockDataSize + j*nt + t]++;
}

int main() {

	time_t sTime = time(NULL);
        struct timeval tt1, tt2;
        int ms;
        double fms;

	int n = 8388608;

	double *data = (double*) malloc(n * sizeof(double));
        for (int i=0; i<n; i++) {
                data[i] = 0;
        }

	double *data_dev;
        cudaMalloc((void**) &data_dev, n * sizeof(double));

	cudaMemcpy(data_dev, data, n * sizeof(double) , cudaMemcpyHostToDevice);
	cudaError_t error = cudaGetLastError();
	cout << "copy to device = " << error << " : " << cudaGetErrorString(error) << endl;
	
	// (*) modify execution parameters
	dim3 nBlocks(1024,1,1);
	dim3 nThreads(256,1,1);

	cudaThreadSynchronize();
        gettimeofday( &tt1, NULL );
        
	// (*) call kernel for coalesced or non-coalesced versions
        //addOne <<< nBlocks, nThreads  >>>(n, data_dev);
	addOne2 <<< nBlocks, nThreads  >>>(n, data_dev);
	error = cudaGetLastError();
        cout << "run kernel = " << error << " : " << cudaGetErrorString(error) << endl;

	cudaThreadSynchronize();
        gettimeofday( &tt2, NULL );

	ms = (tt2.tv_sec - tt1.tv_sec);
        ms = ms * 1000000 + (tt2.tv_usec - tt1.tv_usec);
        fms = ((double)ms)/1000000.0;
        cout << "Comp time = " << fms << endl;

        cudaMemcpy(data, data_dev, n * sizeof(double) , cudaMemcpyDeviceToHost);
	error = cudaGetLastError();
        cout << "copy from device = " << error << " : " << cudaGetErrorString(error) << endl;

	cudaFree(data_dev);

	cout << "data[n-1] = " << data[n-1] << endl;
	free(data);
}



