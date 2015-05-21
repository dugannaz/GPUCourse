#include <iostream>
#include <iomanip>
#include <sys/time.h>
#include <cuda.h>
// (*) include curand device library
#include <curand_kernel.h>

using namespace std;

__global__ void setup_kernel(curandState *state, int init) {

  int id = threadIdx.x + blockIdx.x * blockDim.x;
  // (*) initialize curand generator

  // Each thread gets different seed, same sequence number
  curand_init(id+init, 0, 0, &state[id]);
  
  // Each thread gets same seed, different sequence number
  //curand_init(init, id, 0, &state[id]);	
}

__global__ void walk(curandState *state, double *result) {

	extern __shared__ double smem[];

	int gid = threadIdx.x + blockIdx.x * blockDim.x;
	int id = threadIdx.x;

	// (*) generate uniform random numbers between 0 and 1
	smem[id] = curand_uniform_double(&state[gid]);

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
        struct timeval tt1, tt2, tt3, tt4;
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

	// (*) allocate space for curand states on device 
	curandState *devStates;
	cudaMalloc((void**)&devStates, nBlocks*nThreads*sizeof(curandState));

	cudaThreadSynchronize();
        gettimeofday( &tt1, NULL );

	// (*) run setup kernel
	setup_kernel <<< nBlocks, nThreads >>> (devStates,time(NULL));    

	cudaThreadSynchronize();
        gettimeofday( &tt2, NULL );

	// random walk kernel run 
        walk <<< nBlocks, nThreads, nThreads*sizeof(double) >>> (devStates, devResult);

	cudaThreadSynchronize();
        gettimeofday( &tt3, NULL );

	// transfer results from device
        cudaMemcpy(hostResult, devResult, nBlocks*sizeof(double), cudaMemcpyDeviceToHost);

        // summation of block results
	double total = 0.0;
	for (int i=0; i<nBlocks; i++) {
		total += hostResult[i];
	}

	cudaThreadSynchronize();
        gettimeofday( &tt4, NULL );

	// time calculation
	ms = (tt2.tv_sec - tt1.tv_sec);
        ms = ms * 1000000 + (tt2.tv_usec - tt1.tv_usec);
        fms = ((double)ms)/1000000.0;
        cout << "Initialization time = " << fms << endl;

	ms = (tt3.tv_sec - tt2.tv_sec);
        ms = ms * 1000000 + (tt3.tv_usec - tt2.tv_usec);
        fms = ((double)ms)/1000000.0;
        cout << "Random walk time = " << fms << endl;

	ms = (tt4.tv_sec - tt1.tv_sec);
        ms = ms * 1000000 + (tt4.tv_usec - tt1.tv_usec);
        fms = ((double)ms)/1000000.0;
        cout << "Total time = " << fms << endl;
	
	// screen output of result
	cout << "Total distance = " << setprecision(9) << total << " in " << n << " steps." << endl;
	cout << "Expected distance = " << n/2 << endl;
	// cleanup 
	cudaFree(devResult);
	cudaFree(devStates);
	free(hostResult);
}

