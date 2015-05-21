#include <cuda.h>
#include <iostream>
#include <sys/time.h>

using namespace std;

/* example for thread divergence
 */
__global__ void diverge(int n, double *data) {

	int nt = blockDim.x;

	int b = blockIdx.x;
	int t = threadIdx.x;	

	// data is read to register
	double local = data[b*nt + t];

	// (*) modify if statement for divergence

	if (true) {            // no divergence
	//if (t%32 < 16) {     // divergence in warp execution
	//if (t < 512) {       // divergence not in warp execution
	
		for (int i=0; i<1000; i++)
			local += (double)i*(double)(i+1);

	} else {

		for (int i=1000; i<2000; i++)
			local += (double)i*(double)(i+1);
	}

	// computed result is written back to global memory
	data[b*nt + t] = local;
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
	
	dim3 nBlocks(8192,1,1);
	dim3 nThreads(1024,1,1);

	cudaThreadSynchronize();
        gettimeofday( &tt1, NULL );
        
	diverge <<< nBlocks, nThreads  >>>(n, data_dev);

	cudaThreadSynchronize();
        gettimeofday( &tt2, NULL );

	ms = (tt2.tv_sec - tt1.tv_sec);
        ms = ms * 1000000 + (tt2.tv_usec - tt1.tv_usec);
        fms = ((double)ms)/1000000.0;
        cout << "Comp time = " << fms << endl;

        cudaMemcpy(data, data_dev, n * sizeof(double) , cudaMemcpyDeviceToHost);

	cudaFree(data_dev);

	cout << "data[n-1] = " << data[n-1] << endl;
	free(data);
}



