#include <cuda.h>
#include <iostream>
#include <sys/time.h>

using namespace std;

/* Broadcast example
 */
__global__ void broadcast(const int n, double *data) {

	extern __shared__ double smem[];

	int nt = blockDim.x;
	int t = threadIdx.x;
	int b = blockIdx.x;	

	int nPerThread = 32;

	int i = b*(nt*nPerThread);	

	for (int j=0; j<nPerThread; j++)
		smem[j*nt + t] = data[i + j*nt + t];

	// broadcast : half warp of threads access same data
        // (compare speed with usual access pattern below)
	for (int j=0; j<nPerThread; j++)
		smem[j*nt + t/16]++;

	// usual access pattern
	//for (int j=0; j<nPerThread; j++)
        //        data[j*nt + t]++;

	for (int j=0; j<nPerThread; j++)
		data[i + j*nt + t] = smem[j*nt + t];
	
}

int main() {

	time_t sTime = time(NULL);
        struct timeval tt1, tt2;
        int ms;
        double fms;

	int n = 32*128*256;

	double *data = (double*) malloc(n * sizeof(double));
        for (int i=0; i<n; i++) {
                data[i] = i;
        }

	double *data_dev;
        cudaMalloc((void**) &data_dev, n * sizeof(double));

	cudaMemcpy(data_dev, data, n * sizeof(double) , cudaMemcpyHostToDevice);
	cudaError_t error = cudaGetLastError();
	cout << "copy to device = " << error << " : " << cudaGetErrorString(error) << endl;

	cudaThreadSynchronize();
        gettimeofday( &tt1, NULL );

        int nBlocks = 256;
	int nThreads = 128;
	int sharedMem = nThreads * 32 * sizeof(double);

	broadcast <<< nBlocks, nThreads, sharedMem >>>(n, data_dev);

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

	cout << "data[0] = " << data[n-1] << endl;
	free(data);
}



