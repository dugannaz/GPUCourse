#include <iostream>
#include <sys/time.h>

#define TILE_DIM 32

using namespace std;

/* cudaMallocPitch example:
 * compares normal device memory allocation and allocation using cudaMallocPitch
 * for different matrix sizes.
 * When matrix width is not a multiple of 16 cudaMallocPitch should be preferred.
 */

// kernel for normal device memory allocation
__global__ void matmul(double *a, double* b, double *c, int aw, int bw, int enlarge) {

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	double sum = 0.0;

	for (int i = 0; i < aw-enlarge; i++) {
		sum += a[row*aw+i] * b[i*bw+col];
	}

	c[row*bw+col] = sum;
}

// kernel for allocation with cudaMallocPitch
__global__ void matmul_pitch(double *a, double* b, double *c, int aw, int bw, size_t pitch, int enlarge) {

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	double sum = 0.0;

	for (int i = 0; i < aw-enlarge; i++) {
		sum += a[row*pitch+i] * b[i*pitch+col];
	}

	c[row*pitch+col] = sum;
}

void run_matmul(int ah, int aw, int bw, int enlarge) {

	time_t sTime = time(NULL);
        timeval tt1, tt2;
        int ms;
        double fms;

        int bh=aw;

        int ah1 = ah + enlarge;
        int aw1 = aw + enlarge;
        int bh1 = bh + enlarge;
        int bw1 = bw + enlarge;

	// host arrays

	double *a = (double*)malloc(ah1*aw1*sizeof(double));

	double *b = (double*)malloc(bh1*bw1*sizeof(double));

	double *c = (double*)malloc(ah1*bw1*sizeof(double));

	for (int i=0;i<ah1;i++)
	   for (int j=0;j<aw1;j++)
		a[i*ah1+j] = (double)(i+j);

	for (int i=0;i<bh1;i++)
	   for (int j=0;j<bw1;j++)
		b[i*bh1+j] = (double)(i-j);

	// device arrays

	double *a_dev;
        cudaMalloc((void**) &a_dev, ah1*aw1 * sizeof(double));

	double *b_dev;
        cudaMalloc((void**) &b_dev, bh1*bw1 * sizeof(double));

	double *c_dev;
        cudaMalloc((void**) &c_dev, ah1*bw1 * sizeof(double));

	// copy to device

	cudaMemcpy(a_dev, a, ah1*aw1 * sizeof(double) , cudaMemcpyHostToDevice);

	cudaMemcpy(b_dev, b, bh1*bw1 * sizeof(double) , cudaMemcpyHostToDevice);

	// kernel run

	dim3 nBlocks(bw/TILE_DIM, ah/TILE_DIM, 1);
	dim3 nThreads(TILE_DIM, TILE_DIM, 1);

	cudaThreadSynchronize();
	gettimeofday( &tt1, NULL );

	matmul <<< nBlocks, nThreads >>> (a_dev, b_dev, c_dev, aw1, bw1, enlarge);

	cudaThreadSynchronize();
	gettimeofday( &tt2, NULL );

	// copy from device

	cudaMemcpy(c, c_dev, ah1*bw1 * sizeof(double) , cudaMemcpyDeviceToHost);

	// timing
	cout << "-----------------------------------------------" << endl;
	cout << "normal device memory alocation using cudaMalloc:" << endl;

	ms = (tt2.tv_sec - tt1.tv_sec);
        ms = ms * 1000000 + (tt2.tv_usec - tt1.tv_usec);
        fms = ((double)ms)/1000000.0;
        cout << "Comp time = " << fms << endl;

	double dNumOps = 2.0 * (double)(aw) * (double)(ah) * (double)(bw);
    	double gflops = 1.0e-9 * dNumOps/fms;
	cout << "GFlops = " << gflops << endl;

	cout << "value check = " << c[145] << endl;
	cout << "-----------------------------------------------" << endl;

	free(a);
        free(b);
        free(c);
        cudaFree(a_dev);
        cudaFree(b_dev);
        cudaFree(c_dev);

}

void run_matmul_pitch(int ah, int aw, int bw, int enlarge) {

	time_t sTime = time(NULL);
        timeval tt1, tt2;
        int ms;
        double fms;

	int bh=aw;

	int ah1 = ah + enlarge;
        int aw1 = aw + enlarge;
        int bh1 = bh + enlarge;
        int bw1 = bw + enlarge;


	// host arrays

	double *a = (double*)malloc(ah1*aw1*sizeof(double));

	double *b = (double*)malloc(bh1*bw1*sizeof(double));

	double *c = (double*)malloc(ah1*bw1*sizeof(double));

	for (int i=0;i<ah1;i++)
	   for (int j=0;j<aw1;j++)
		a[i*ah1+j] = (double)(i+j);

	for (int i=0;i<bh1;i++)
	   for (int j=0;j<bw1;j++)
		b[i*bh1+j] = (double)(i-j);

	// device arrays are allocated using cudaMallocPitch

	size_t pitch;

	double *a_dev;
	cudaMallocPitch(&a_dev, &pitch, aw1 * sizeof(double), ah1);

	double *b_dev;
	cudaMallocPitch(&b_dev, &pitch, bw1 * sizeof(double), bh1);

	double *c_dev;
	cudaMallocPitch(&c_dev, &pitch, bw1 * sizeof(double), ah1);

	// data is copied with cudaMemcpy2D
	cudaMemcpy2D(a_dev, pitch, a, aw1 * sizeof(double), aw1, ah1, cudaMemcpyHostToDevice);
	cudaMemcpy2D(b_dev, pitch, b, bw1 * sizeof(double), bw1, bh1, cudaMemcpyHostToDevice);

	// kernel run

	dim3 nBlocks(bw/TILE_DIM, ah/TILE_DIM, 1);
	dim3 nThreads(TILE_DIM, TILE_DIM, 1);

	cudaThreadSynchronize();
	gettimeofday( &tt1, NULL );

	matmul_pitch <<< nBlocks, nThreads >>> (a_dev, b_dev, c_dev, aw1, bw1, 
						pitch/sizeof(double),enlarge);


	cudaThreadSynchronize();
	gettimeofday( &tt2, NULL );

	// data is copied back with cudaMemcpy2D

	cudaMemcpy2D(c, bw1 * sizeof(double), c_dev, pitch, bw1, ah1, cudaMemcpyDeviceToHost);

	// timing

	cout << "-----------------------------------------------" << endl;
        cout << "device memory alocation using cudaMallocPitch:" << endl;
	ms = (tt2.tv_sec - tt1.tv_sec);
        ms = ms * 1000000 + (tt2.tv_usec - tt1.tv_usec);
        fms = ((double)ms)/1000000.0;
        cout << "Comp time = " << fms << endl;

	double dNumOps = 2.0 * (double)(aw) * (double)(ah) * (double)(bw);
    	double gflops = 1.0e-9 * dNumOps/fms;
	cout << "GFlops = " << gflops << endl;

	cout << "value check = " << c[145] << endl;
	cout << "-----------------------------------------------" << endl;

	free(a);
	free(b);
	free(c);
	cudaFree(a_dev);
	cudaFree(b_dev);
	cudaFree(c_dev);
}


int main() {

	int ah=2560;
        int aw=2560;
        int bw=2560;

	// enlarges matrix dimensions given amount
	// calculation is carried out over original matrix dimensions
	// (*) give different values to see the effect on computation 
	//     times of normal memory allocation and using cudaMallocPitch
	int enlarge = 0;

	run_matmul(ah, aw, bw, enlarge);
	run_matmul_pitch(ah, aw, bw, enlarge);
}

