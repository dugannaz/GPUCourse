#include <iostream>
#include "mkl.h"
#include <sys/time.h>

// compile: -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread

/* Intel MKL Library CUBLAS double precision matrix multiplication
 * (For performance comparison with GPU matrix multiplication)   
 */
using namespace std;

int main() {

	struct timeval t1, t2;
	double dt1, dt2;

	int n = 2560;
	int lda,ldb,ldc;
	lda = ldb = ldc = n;

	char *N1 = "N";
	char *N2 = "N";
	double one = 1.0;
	double zer = 0.0;


	double *a = (double*)malloc(n*n*sizeof(double));

	double *b = (double*)malloc(n*n*sizeof(double));

	for (int i=0;i<n;i++)
	   for (int j=0;j<n;j++)
		a[i+j*n] = (double)(i+j);

	for (int i=0;i<n;i++)
	   for (int j=0;j<n;j++)
		b[i+j*n] = (double)(i-j);

	double *c = (double*)malloc(n*n*sizeof(double));

	gettimeofday(&t1, NULL);

	dgemm(N1,N2,&n,&n,&n,&one,a,&lda,b,&ldb,&zer,c,&ldc);

	gettimeofday(&t2, NULL);

	dt2 = t2.tv_sec - t1.tv_sec + 1.0e-6 * (t2.tv_usec - t1.tv_usec);
	cout << "Comp time = " << dt2 << endl;

	double dNumOps = 2.0 * (double)n * (double)n * (double)n;
    	double gflops = 1.0e-9 * dNumOps/dt2;
	cout << "GFlops = " << gflops << endl;

	cout << c[999*n+145] << " " << c[567*n+232] << endl;
} 
