#include <iostream>
#include <sys/time.h>

#define TILE_DIM 32

using namespace std;

/* Compile with "-Xptxas -dlcm=cg" flags to disable Fermi L1 cache.
 * Code would slow down when L1 cache is disabled.
 * Disabling L1 cache would not have any effect on the shared memory 
 *    version of matmul program (see exercises).
 */

__global__ void matmul(double *a, double* b, double *c, int aw, int bw) {

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	double sum = 0.0;

	for (int i = 0; i < aw; i++) {
		sum += a[row*aw+i] * b[i*bw+col];
	}

	c[row*bw+col] = sum;
}

int main() {

	time_t sTime = time(NULL);
        timeval tt1, tt2;
        int ms;
        double fms;

	int ah=2560;
	int aw=2560;
	int bh=2560;
	int bw=2560;


	double *a = (double*)malloc(ah*aw*sizeof(double));

	double *b = (double*)malloc(bh*bw*sizeof(double));

	double *c = (double*)malloc(ah*bw*sizeof(double));

	for (int i=0;i<ah;i++)
	   for (int j=0;j<aw;j++)
		a[i*ah+j] = (double)(i+j);

	for (int i=0;i<bh;i++)
	   for (int j=0;j<bw;j++)
		b[i*bh+j] = (double)(i-j);


	double *a_dev;
        cudaMalloc((void**) &a_dev, ah*aw * sizeof(double));

	double *b_dev;
        cudaMalloc((void**) &b_dev, bh*bw * sizeof(double));

	double *c_dev;
        cudaMalloc((void**) &c_dev, ah*bw * sizeof(double));


	cudaMemcpy(a_dev, a, ah*aw * sizeof(double) , cudaMemcpyHostToDevice);

	cudaMemcpy(b_dev, b, bh*bw * sizeof(double) , cudaMemcpyHostToDevice);


	dim3 nBlocks(bw/TILE_DIM, ah/TILE_DIM, 1);
	dim3 nThreads(TILE_DIM, TILE_DIM, 1);

	// (*) Set shared mem size 16KB, L1 cache size 48 KB
        cudaFuncSetCacheConfig(matmul, cudaFuncCachePreferL1);

	gettimeofday( &tt1, NULL );

	matmul <<< nBlocks, nThreads >>> (a_dev, b_dev, c_dev, aw, bw);

	cudaThreadSynchronize();
	gettimeofday( &tt2, NULL );


	cudaMemcpy(c, c_dev, ah*bw * sizeof(double) , cudaMemcpyDeviceToHost);


	ms = (tt2.tv_sec - tt1.tv_sec);
        ms = ms * 1000000 + (tt2.tv_usec - tt1.tv_usec);
        fms = ((double)ms)/1000000.0;
        cout << "Comp time = " << fms << endl;

	double dNumOps = 2.0 * (double)aw * (double)aw * (double)bw;
    	double gflops = 1.0e-9 * dNumOps/fms;
	cout << "GFlops = " << gflops << endl;

	cout << "value check = " << c[145] << endl;
}



