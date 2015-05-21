#include <iostream>
#include <sys/time.h>

using namespace std;

#define TILE_DIM 32

/* Matrix multiplication
 * - shared memory with tile
 * - double - single precision
 * - gpu - cpu comparison
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

__global__ void matmul_shared(double *a, double* b, double *c, int aw, int bw) {

	// (*) create 2D shared memory arrays of size (TILE_DIM x TILE_DIM) for matrices (a) and (b)
	

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	double sum = 0.0;

	// This loop is necessary since (aw) and (bh) are not equal to TILE_DIM but a multiply of it 
	for (int ia=0; ia<aw; ia+=TILE_DIM) {

		// copy block data of iteration (ia) to shared memory
		aTile[threadIdx.y][threadIdx.x] = a[row*aw + ia + threadIdx.x];
		bTile[threadIdx.y][threadIdx.x] = b[(ia+threadIdx.y)*bw+col];

		// (*) synchronize if necessary
		

		// (*) do multiplication
		for (int i = 0; i < TILE_DIM; i++) {
			sum += /* ?????????????? */
		}

		// (*) synchronize if necessary
		
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
	int bh=aw;
	int bw=2560;

	// host arrays

	double *a = (double*)malloc(ah*aw*sizeof(double));

	double *b = (double*)malloc(bh*bw*sizeof(double));

	double *c = (double*)malloc(ah*bw*sizeof(double));

	for (int i=0;i<ah;i++)
	   for (int j=0;j<aw;j++)
		a[i*ah+j] = (double)(i+j);

	for (int i=0;i<bh;i++)
	   for (int j=0;j<bw;j++)
		b[i*bh+j] = (double)(i-j);

	// device arrays

	double *a_dev;
        cudaMalloc((void**) &a_dev, ah*aw * sizeof(double));

	double *b_dev;
        cudaMalloc((void**) &b_dev, bh*bw * sizeof(double));

	double *c_dev;
        cudaMalloc((void**) &c_dev, ah*bw * sizeof(double));

	// copy to device

	cudaMemcpy(a_dev, a, ah*aw * sizeof(double) , cudaMemcpyHostToDevice);

	cudaMemcpy(b_dev, b, bh*bw * sizeof(double) , cudaMemcpyHostToDevice);

	// kernel run

	dim3 nBlocks(bw/TILE_DIM, ah/TILE_DIM, 1);
	dim3 nThreads(TILE_DIM, TILE_DIM, 1);

	gettimeofday( &tt1, NULL );

	//matmul <<< nBlocks, nThreads >>> (a_dev, b_dev, c_dev, aw, bw);
	matmul_shared <<< nBlocks, nThreads >>> (a_dev, b_dev, c_dev, aw, bw);

	cudaThreadSynchronize();
	gettimeofday( &tt2, NULL );

	// copy from device

	cudaMemcpy(c, c_dev, ah*bw * sizeof(double) , cudaMemcpyDeviceToHost);

	// timing

	ms = (tt2.tv_sec - tt1.tv_sec);
        ms = ms * 1000000 + (tt2.tv_usec - tt1.tv_usec);
        fms = ((double)ms)/1000000.0;
        cout << "Comp time = " << fms << endl;

	double dNumOps = 2.0 * (double)aw * (double)aw * (double)bw;
    	double gflops = 1.0e-9 * dNumOps/fms;
	cout << "GFlops = " << gflops << endl;

        cout << c[145*bw+999] << " " << c[232*bw+567] << endl;
}



