#include <iostream>
#include <cuda.h>
#include <sys/time.h>

using namespace std;

__global__ void reduction(double *data, double *result) {

	int gid = threadIdx.x + blockIdx.x * blockDim.x *2;
	int id = threadIdx.x;

	for(int s=blockDim.x; s>0; s>>=1) {
        	if (id < s) {
            		data[gid] += data[gid + s];
        	}
        	__syncthreads();
    	}

	if (id == 0) 
		result[blockIdx.x] = data[gid];
}

__global__ void reductionShared(double *data, double *result) {

	// (*) define dynamic shared memory
	extern __shared__ double smem[];

	int gid = threadIdx.x + blockIdx.x * blockDim.x *2;
	int id = threadIdx.x;

	// (*) copy block data to shared memory
	smem[id] = data[gid];
	smem[ blockDim.x+id] = data[blockDim.x+gid];

	// (*) synchronize threads
	__syncthreads();

	// (*) do reduction in shared memory
	for(int s=blockDim.x; s>0; s>>=1) {
        	if (id < s) {
            		smem[id] += smem[id + s];
        	}
        	__syncthreads();
    	}

	// (*) copy block result to result array
	if (id == 0) 
		result[blockIdx.x] = smem[id];
}

int main(int argc, char *argv[]) {

	// time variables
	time_t sTime = time(NULL);
        struct timeval tt1, tt2;
        int ms;
        double fms;

	// data size
	int n = 4194304; // (= 2^22)

	// (*) set block size to its maximum value
	int nThreads = 1024;
	int nBlocks = n/(2*nThreads);
	
	// data and result arrays
	double *devData, *hostData;
	double *devResult, *hostResult;

	// host memory allocation 
        hostData = (double*)malloc(n*sizeof(double));
	hostResult = (double*)malloc(nBlocks*sizeof(double));

	// data initializtion
	for (int i=0; i<n; i++)
		hostData[i] = (double)i;

	// device memory allocation
        cudaMalloc((void**)&devData, n*sizeof(double));
	cudaMalloc((void**)&devResult, nBlocks*sizeof(double));

	// copy data to device
	cudaMemcpy(devData, hostData, n*sizeof(double), cudaMemcpyHostToDevice);	

	// kernel run with timing
	cudaThreadSynchronize();
        gettimeofday( &tt1, NULL );

	// (*) modify kernel run for shared memory version
	//reduction <<< nBlocks, nThreads >>> (devData, devResult);
        reductionShared <<< nBlocks, nThreads, 2*nThreads*sizeof(double) >>> (devData, devResult);
	

	cudaThreadSynchronize();
        gettimeofday( &tt2, NULL );

	// transfer results from device
        cudaMemcpy(hostResult, devResult, nBlocks*sizeof(double), cudaMemcpyDeviceToHost);

        // summation of block results
	double total = 0;
	for (int i=0; i<nBlocks; i++) {
		total += hostResult[i];
	}
	
	// screen output of result
	cout << "Total = " << total << endl;

	// time calculation
	ms = (tt2.tv_sec - tt1.tv_sec);
        ms = ms * 1000000 + (tt2.tv_usec - tt1.tv_usec);
        fms = ((double)ms)/1000000.0;
        cout << "Comp time = " << fms << endl;

	// cleanup 
        cudaFree(devData);
	cudaFree(devResult);
	free(hostData);
	free(hostResult);
}

