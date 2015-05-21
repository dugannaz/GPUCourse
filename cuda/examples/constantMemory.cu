//Consant memeory 64KB max, 2^16 bytes, 2^14 integers, 16384

#include <stdio.h>
#include <cuda.h>

// size of vectors, has to be known at compile time, max available
#define N 8192         

// Constants held in constant memory

__device__ __constant__ int dev_a_Cont[N];
__device__ __constant__ int dev_b_Cont[N];

// regular global memory for comparison

__device__  int dev_a[N];
__device__  int dev_b[N];

// result in device global memory

__device__ int dev_c[N]; //device global memory for result

// kernel routines

__global__ void add_Cont() {
	int tid = blockIdx.x *  blockDim.x + threadIdx.x;
        if(tid < N){
        	dev_c[tid] = dev_a_Cont[tid] + dev_b_Cont[tid];
        }
}

__global__ void add() {
	int tid = blockIdx.x *  blockDim.x + threadIdx.x;
        if(tid < N){
        	dev_c[tid] = dev_a[tid] + dev_b[tid];
        }
}


int main()  {
	
	// threads per block and blocks per grid
	int T = 128, B = 64;           	 
	int a[N],b[N],c[N]; //statically declared host vectors

	cudaEvent_t start, stop;  // cuda events to measure time
	float elapsed_time,elapsed_time_Cont; 

	cudaEventCreate(&start); // timing objects
	cudaEventCreate(&stop);

/*----------- GPU not using constant memory ------------------------*/

	printf("GPU not using constant memory\n");

	for(int i=0;i<N;i++) {   // load arrays with some numbers
		a[i] = i;
		b[i] = i*2;
	}

	// copy vectors to constant memory
	cudaMemcpyToSymbol(dev_a,a,N*sizeof(int),0,cudaMemcpyHostToDevice); 
	cudaMemcpyToSymbol(dev_b,b,N*sizeof(int),0,cudaMemcpyHostToDevice);

	cudaEventRecord(start, 0); // start time

	add<<<B,T>>>();   // does not need array ptrs now

	cudaThreadSynchronize(); // wait for all threads to complete

	cudaEventRecord(stop, 0); // instrument code to measure end time

	cudaMemcpyFromSymbol(a,dev_a,N*sizeof(int),0,cudaMemcpyDeviceToHost);
	cudaMemcpyFromSymbol(b,dev_b,N*sizeof(int),0,cudaMemcpyDeviceToHost);
	cudaMemcpyFromSymbol(c,dev_c,N*sizeof(int),0,cudaMemcpyDeviceToHost);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);

	printf("Checking results\n");
	for(int i=0;i<N;i++) {
	   if (a[i] + b[i] != c[i]) {
		printf("ERROR IN COMPUTATION\n");
		break;
	   }
	}

	// print out execution time
	printf("Time to calculate results: %f ms.\n", elapsed_time);  

/*----------- GPU using constant memory ------------------------*/

	printf("GPU using constant memory\n");

	for(int i=0;i<N;i++) {   // load arrays with some numbers
		a[i] = i;
		b[i] = i*2;
	}

	// copy vectors to constant memory

	cudaMemcpyToSymbol(dev_a_Cont,a,N*sizeof(int),0,cudaMemcpyHostToDevice); 	
	cudaMemcpyToSymbol(dev_b_Cont,b,N*sizeof(int),0,cudaMemcpyHostToDevice);

	cudaEventRecord(start, 0);		// start time

	add_Cont<<<B,T>>>();			// does not need array ptrs now

	cudaThreadSynchronize();		// wait for all threads to complete

	cudaEventRecord(stop, 0);     	// instrument code to measure end time

	cudaMemcpyFromSymbol(a,dev_a_Cont,N*sizeof(int),0,cudaMemcpyDeviceToHost);
	cudaMemcpyFromSymbol(b,dev_b_Cont,N*sizeof(int),0,cudaMemcpyDeviceToHost);
	cudaMemcpyFromSymbol(c,dev_c,N*sizeof(int),0,cudaMemcpyDeviceToHost);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time_Cont, start, stop);

	printf("Checking results\n");
	for(int i=0;i<N;i++) {
	   if (a[i] + b[i] != c[i]) {
		printf("ERROR IN COMPUTATION\n");
		break;
	   }
	}

	// print out execution time
	printf("Time to calculate results: %f ms.\n", elapsed_time_Cont);  

	printf("Speedup using constant memory = %f\n",elapsed_time/elapsed_time_Cont);



/* ----------- clean up, no malloc free needed ---------*/
	
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return 0;
}
