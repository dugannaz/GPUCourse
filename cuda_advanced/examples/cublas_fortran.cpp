#include <iostream>
#include <stdlib.h>
#include <sys/time.h>
#include <cublas.h>

using namespace std;

extern "C" void cublasdgemm_(int *aT, int *bT, int *na_p, int *nb_p, int *nc_p, 
			     double *alpha, double *a, int *lda_p, double *b, int *ldb_p,
			     double *beta, double *c, int *ldc_p) {

	struct timeval t1, t2;
        double dt, flops;

	int na = *na_p;
	int nb = *nb_p;
	int nc = *nc_p;

	char trans_a = 'N'; 
	char trans_b = 'N';

	if (*aT == 1) trans_a='T';
	if (*bT == 1) trans_b='T';
       
	// variable definitions for matrices
	double *d_a, *d_b, *d_c;	
	int lda, ldb, ldc;

	lda = *lda_p; ldb = *ldb_p; ldc = *ldc_p;

	// initialize cublas
	cublasInit();

	// allocate device memory for matrices a,b,c
	cudaMalloc((void**)&d_a, na*lda * sizeof(double));
	cudaMalloc((void**)&d_b, nb*ldb * sizeof(double));
	cudaMalloc((void**)&d_c, nc*ldc * sizeof(double));

	// copy matrices a and b to device using cublasSetMatrix
	cublasSetMatrix(lda,na,sizeof(double),a,lda,d_a,lda);
	cublasSetMatrix(ldb,nb,sizeof(double),b,ldb,d_b,lda);

	cudaThreadSynchronize();
	gettimeofday(&t1, NULL);

	// run matrix multiplication
	cublasDgemm(trans_a,trans_b,na,nb,nc,*alpha,d_a,lda,d_b,ldb,*beta,d_c,ldc);

	cudaThreadSynchronize();
	gettimeofday(&t2, NULL);

	// copy result matrix c to host using cublasGetMAtrix
	cublasGetMatrix(ldc,nc,sizeof(double),d_c,ldc,c,ldc);

	// deallocate device memory
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	// timing
	dt = (t2.tv_sec - t1.tv_sec + 1.0e-6 * (t2.tv_usec - t1.tv_usec));
	flops = 2.0 * double(na) * double(nb) * double(nc);
	
	cout << "Computation Time = " << dt << endl;
	cout << "GFlops/sec = " << 1.0e-9*flops/dt << endl;

	// shutdown cublas
	cublasShutdown();
}

