#include <iostream>
#include <iomanip>
#include <sys/time.h>
#include <cuda.h>
// (*) include curand device library


using namespace std;

__global__ void setup_kernel(curandState *state) {

  int id = threadIdx.x + blockIdx.x * blockDim.x;
  // (*) initialize curand generator
  
}

__global__ void walk(curandState *state, double *result) {

	extern __shared__ double smem[];

	int gid = threadIdx.x + blockIdx.x * blockDim.x;
	int id = threadIdx.x;

	// (*) generate double precision uniform random numbers in [0,1]
	

	__syncthreads();

	// reduction in shared memory
	for(int s=blockDim.x/2; s>0; s>>=1) {
        	if (id < s) {
            		smem[id] += smem[id + s];
        	}
        	__syncthreads();
    	}

	// copy block result to result array
	if (id == 0) 
		result[blockIdx.x] = smem[id];
}

int main(int argc, char *argv[]) {

	// time variables
	time_t sTime = time(NULL);
        struct timeval tt1, tt2, tt3;
        int ms;
        double fms;

	// number of steps
	int n = 1048576;

	// runtime configuration parameters 
	int nThreads = 1024;
	int nBlocks = n/nThreads;
	
	// data and result arrays
	double *devResult, *hostResult;

	// host memory allocation 
	hostResult = (double*)calloc(nBlocks,sizeof(double));

	// device memory allocation
	cudaMalloc((void**)&devResult, nBlocks*sizeof(double));

	curandState *devStates;
	// (*) allocate space for curand states on device 


	cudaThreadSynchronize();
        gettimeofday( &tt1, NULL );	

	// (*) run setup kernel
	  
	cudaThreadSynchronize();
        gettimeofday( &tt2, NULL );

	// random walk kernel run  
        walk <<< nBlocks, nThreads, nThreads*sizeof(double) >>> (devStates, devResult);

	cudaThreadSynchronize();
        gettimeofday( &tt3, NULL );

	// time calculation
	ms = (tt2.tv_sec - tt1.tv_sec);
        ms = ms * 1000000 + (tt2.tv_usec - tt1.tv_usec);
        fms = ((double)ms)/1000000.0;
        cout << "Initialization time = " << fms << endl;

	ms = (tt3.tv_sec - tt2.tv_sec);
        ms = ms * 1000000 + (tt3.tv_usec - tt2.tv_usec);
        fms = ((double)ms)/1000000.0;
        cout << "Random walk time = " << fms << endl;

	// transfer results from device
        cudaMemcpy(hostResult, devResult, nBlocks*sizeof(double), cudaMemcpyDeviceToHost);

        // summation of block results
	double total = 0.0;
	for (int i=0; i<nBlocks; i++) {
		total += hostResult[i];
	}
	
	// screen output of result
	cout << "Total distance = " << setprecision(9) << total << " in " << n << " steps." << endl;
	cout << "Expected distance = " << n/2 << endl;
	// cleanup 
	cudaFree(devResult);
	cudaFree(devStates);
	free(hostResult);
}

