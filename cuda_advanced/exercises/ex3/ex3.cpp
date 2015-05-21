#include <iostream>
#include <stdlib.h>
#include <sys/time.h>
// (*) include cublas library

/* CUBLAS matrix multiplication
 * - cublas usage
 * - compare with cpu for single and double precision
 */

using namespace std;

int main() {

        struct timeval t1, t2;
        double dt, flops;

	// variable definitions for matrices
	double *a, *b, *c;
	double *d_a, *d_b, *d_c;	
	int lda, ldb, ldc;

	// square matrix side length
	int n = 2560;

	lda = ldb = ldc = n;

	// (*) initialize cublas here
	

	// host memory allocation
	cudaMallocHost((void**) &a, n*lda*sizeof(double));
	cudaMallocHost((void**) &b, n*ldb*sizeof(double));
	cudaMallocHost((void**) &c, n*ldc*sizeof(double));

	// data initialization
	for (int i=0;i<n;i++)
	   for (int j=0;j<n;j++)
		a[i+j*lda] = (double)(i+j);

	for (int i=0;i<n;i++)
	   for (int j=0;j<n;j++)
		b[i+j*ldb] = (double)(i-j);

	// (*) allocate device memory for matrices a,b,c
	

	// (*) copy matrices a and b to device using cublasSetMatrix
	

	cudaThreadSynchronize();
	gettimeofday(&t1, NULL);

	// (*) run matrix multiplication 
		

	cudaThreadSynchronize();
	gettimeofday(&t2, NULL);

	// (*) copy result matrix c to host using cublasGetMatrix
	

	// (*) deallocate device memory
	

        cout << c[999*ldc+145] << " " << c[567*ldc+232] << endl;

	// deallocate host memory
	cudaFreeHost(a);
	cudaFreeHost(b);
	cudaFreeHost(c);


	// timing
	dt = (t2.tv_sec - t1.tv_sec + 1.0e-6 * (t2.tv_usec - t1.tv_usec));
	flops = 2.0 * double(n) * double(n) * double(n);
	
	cout << "Computation Time = " << dt << endl;
	cout << "GFlops/sec = " << 1.0e-9*flops/dt << endl;

	// (*) shutdown cublas
	
}

