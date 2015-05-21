#include <stdio.h>
#include <iostream>
#include <sys/time.h>
#include <cuda.h>
#include <curand.h>

using namespace std;

int main(int argc, char *argv[])
{
        int n = 1048576;

	// (*) define curand generator 


	double *devData, *hostData;

	// time variables
        time_t sTime = time(NULL);
        struct timeval tt1, tt2;
        int ms;
        double fms;

	// host memory allocation 
        hostData = (double*)calloc(n, sizeof(double));

	// GPU memory allocation 
        cudaMalloc((void**)&devData, n*sizeof(double));

	cudaThreadSynchronize();
        gettimeofday( &tt1, NULL );	

	// (*) Create pseudo-random number generator


	// (*) Set seed 

	
	// (*) Generate (n) double precision random numbers on device 


	// transfer results from GPU memory 
        cudaMemcpy(hostData, devData, n*sizeof(double), cudaMemcpyDeviceToHost);

	// calculate walk distance
	double dist = 0.0;
	for (int i=0; i<n; i++)
     		dist += hostData[i];
		
	cudaThreadSynchronize();
        gettimeofday( &tt2, NULL );

	// timing
	ms = (tt2.tv_sec - tt1.tv_sec);
        ms = ms * 1000000 + (tt2.tv_usec - tt1.tv_usec);
        fms = ((double)ms)/1000000.0;
        cout << "Total time = " << fms << endl;

	cout << "Total distance = " << dist << " in " << n << " steps." << endl;
        cout << "Expected distance = " << n/2 << endl;

	// cleanup 
	curandDestroyGenerator(gen);
        cudaFree(devData);
	free(hostData);

}

