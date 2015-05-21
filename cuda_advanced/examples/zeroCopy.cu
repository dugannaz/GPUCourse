#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include <sys/time.h>

using namespace std;

/* Zero copy example
 * - No need to transfer data to device
 * - Performance is better than usual pinned memory if 
 *   data is read only once in the kernel 
 * - Computation amount in the kernel also affects performance 
 */

__global__ void addOne(int n, double *data) {

	int nb = gridDim.x;
	int nt = blockDim.x;

	int compPerThread = n / (nb*nt);
	int blockDataSize = compPerThread*nt;

	int b = blockIdx.x;
	int t = threadIdx.x;	

	for (int j=0; j<compPerThread; j++) {
	   double local = data[b*blockDataSize + j*nt + t];
	   // this loop is for making a compute intensive kernel
           // zero copy performance will be different for different loop sizes
	   // (*) try with loop sizes = 1, 100 ,10000
	   for (int k=0; k<100; k++)
		local++;
	   data[b*blockDataSize + j*nt + t] = local;
	}
}

int main() {

	// check if device can map host memory
	int deviceCount; 
	cudaGetDeviceCount(&deviceCount); 
	int device; 
	for (device = 0; device < deviceCount; ++device) { 
		cudaDeviceProp deviceProp; 
		cudaGetDeviceProperties(&deviceProp, device); 
		printf("Device %d has compute capability %d.%d.\n", device, 
			deviceProp.major, deviceProp.minor);
		cout << " canMapHostMemory = " << deviceProp.canMapHostMemory << endl;
	}
	/************************/

	time_t sTime = time(NULL);
        struct timeval tt1, tt2;
        int ms;
        double fms;

	int n = 8388608;

	// pinned memory is allocated with special flag for device host mapping
	double *data;
	cudaHostAlloc((void**)&data, n * sizeof(double), cudaHostAllocWriteCombined|cudaHostAllocMapped);
	// device pointer is mapped to host pointer
	double *data_dev;
	cudaHostGetDevicePointer( &data_dev, data, 0);

	// normal pinned memory is allocated
	double *data1;
	cudaMallocHost((void**) &data1, n * sizeof(double));

        for (int i=0; i<n; i++) {
                data[i] = 0.0;
        }

	for (int i=0; i<n; i++) {
                data1[i] = 0.0;
        }

	double *data_dev1;
        cudaMalloc((void**) &data_dev1, n * sizeof(double));

	dim3 nBlocks(1024,1,1);
	dim3 nThreads(256,1,1);

	// host mapped pinned memory :
	// timing for kernel execution (data transfer is included)
	cudaThreadSynchronize();
        gettimeofday( &tt1, NULL );

	addOne <<< nBlocks, nThreads  >>>(n, data_dev);

	cudaThreadSynchronize();
        gettimeofday( &tt2, NULL );

	ms = (tt2.tv_sec - tt1.tv_sec);
        ms = ms * 1000000 + (tt2.tv_usec - tt1.tv_usec);
        fms = ((double)ms)/1000000.0;
        cout << "Host mapped pinned memory : elapsed Time = " << fms << endl;

	cout << "data[n-1] = " << data[n-1] << endl;
	

	// pinned memory :
	// timing for data transfer, kernel execution, data transfer back
	cudaThreadSynchronize();
        gettimeofday( &tt1, NULL );
        
	cudaMemcpy(data_dev1, data1, n * sizeof(double) , cudaMemcpyHostToDevice);

	addOne <<< nBlocks, nThreads  >>>(n, data_dev1);

        cudaMemcpy(data1, data_dev1, n * sizeof(double) , cudaMemcpyDeviceToHost);

	cudaThreadSynchronize();
        gettimeofday( &tt2, NULL );

	ms = (tt2.tv_sec - tt1.tv_sec);
        ms = ms * 1000000 + (tt2.tv_usec - tt1.tv_usec);
        fms = ((double)ms)/1000000.0;
        cout << "Pinned memory : elapsed Time = " << fms << endl;

	cout << "data1[n-1] = " << data1[n-1] << endl;

	cudaFree(data_dev);

	// pinned memory is freed
	cudaFreeHost(data);
	cudaFreeHost(data1);
}



