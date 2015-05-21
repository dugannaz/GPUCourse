#include <iostream>
#include <sys/time.h>
#include <omp.h>

#define TILE_DIM 32

using namespace std;

/* Multiple GPUs
 * - distribute GPUs to OpenMP threads
 * - distribute data (computation) to GPUs
 * - (export CUDA_VISIBLE_DEVICES=1,2)
 */

__global__ void matmul_shared(double *a, double* b, double *c, int aw, int bw) {

	__shared__ double aTile[TILE_DIM][TILE_DIM], bTile[TILE_DIM][TILE_DIM];

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	double sum = 0.0;

	for (int ia=0; ia<aw; ia+=TILE_DIM) {

		aTile[threadIdx.y][threadIdx.x] = a[row*aw + ia + threadIdx.x];
		bTile[threadIdx.y][threadIdx.x] = b[(ia+threadIdx.y)*bw+col];

		__syncthreads();

		for (int i = 0; i < TILE_DIM; i++) {
			sum += aTile[threadIdx.y][i]* bTile[i][threadIdx.x];
		}

		__syncthreads();
	}

	c[row*bw+col] = sum;
}

int main() {

	//---------------------------------------------------------------

	// number of gpus
	int num_gpus = 0;

	// (*) get number of devices using cudaGetDeviceCount


	cout << "number of host CPUs : " << omp_get_num_procs() << endl;
    	cout << "number of CUDA devices: " << num_gpus << endl;

	// set number of threads
	omp_set_num_threads(num_gpus);

	//---------------------------------------------------------------

	// number of matrix multiplication operations
	int nMultiply = 10;

	// work distribution to devices (two devices assumed)
        // (*) find optimum distribution
	int nMultiplyThread[] = {5,5};
	

	time_t sTime = time(NULL);
        timeval tt1[num_gpus], tt2[num_gpus];
        int ms[num_gpus];
        double fms[num_gpus];

	int ah=2560;
	int aw=2560;
	int bh=aw;
	int bw=2560;

	// host arrays

	double *a = (double*)malloc(nMultiply * ah*aw*sizeof(double));

	double *b = (double*)malloc(nMultiply * bh*bw*sizeof(double));

	double *c = (double*)malloc(nMultiply * ah*bw*sizeof(double));

	for (int k=0;k<nMultiply;k++) {

		for (int i=0;i<ah;i++)
		   for (int j=0;j<aw;j++)
			a[i*ah+j+ k*ah*aw] = (double)(i+j)*(k+1);

		for (int i=0;i<bh;i++)
		   for (int j=0;j<bw;j++)
			b[i*bh+j + k*bh*bw] = (double)(i-j)*(k+1);
	}

#pragma omp parallel
{	
	// get thread id
	int cpu_thread_id = omp_get_thread_num();

        int gpu_id = -1;
	
	// (*) set GPU device for this CPU thread
      
 
        cudaGetDevice(&gpu_id);
 	cout << "Thread : " << cpu_thread_id << " uses GPU # " << gpu_id << endl;

	// number of multiplies for this CPU thread
	int nMultiply_thread = nMultiplyThread[cpu_thread_id];

	// device arrays

	double *a_dev;
        cudaMalloc((void**) &a_dev, nMultiply_thread * ah*aw * sizeof(double));

	double *b_dev;
        cudaMalloc((void**) &b_dev, nMultiply_thread * bh*bw * sizeof(double));

	double *c_dev;
        cudaMalloc((void**) &c_dev, nMultiply_thread * ah*bw * sizeof(double));

	// copy to device

	// each thread calculates its host data offset
	int previous = 0;
	for (int pt=0; pt<cpu_thread_id; pt++)
		previous += nMultiplyThread[pt];

	int offsetA = previous * ah*aw;
	int offsetB = previous * bh*bw;
	int offsetC = previous * ah*bw;

	cudaThreadSynchronize();
	gettimeofday( &tt1[cpu_thread_id], NULL );

	// (*) fill host pointers
	cudaMemcpy(a_dev, /*******/, nMultiply_thread * ah*aw * sizeof(double) , cudaMemcpyHostToDevice);
	cudaMemcpy(b_dev, /*******/, nMultiply_thread * bh*bw * sizeof(double) , cudaMemcpyHostToDevice);

	// kernel run

	dim3 nBlocks(bw/TILE_DIM, ah/TILE_DIM, 1);
	dim3 nThreads(TILE_DIM, TILE_DIM, 1);

	for (int n=0; n<nMultiply_thread; n++) {

		matmul_shared <<< nBlocks, nThreads >>> (a_dev + n*ah*aw, b_dev + n*bh*bw, 
							 c_dev + n*ah*bw, aw, bw);
	}

	// copy from device

	// (*) fill host pointer
	cudaMemcpy(/*******/, c_dev, nMultiply_thread * ah*bw * sizeof(double) , cudaMemcpyDeviceToHost);

	cudaThreadSynchronize();
	gettimeofday( &tt2[cpu_thread_id], NULL );
}

	// timing

	for (int i=0; i<num_gpus; i++) {
		ms[i] = (tt2[i].tv_sec - tt1[i].tv_sec);
		ms[i] = ms[i] * 1000000 + (tt2[i].tv_usec - tt1[i].tv_usec);
		fms[i] = ((double)ms[i])/1000000.0;
		cout << "Thread : " << i << " computed " << nMultiplyThread[i]
		     << " matrix multiplications  : Comp time = " << fms[i] << endl;
	}

	cout << "value check = " << c[145] << endl;
}



