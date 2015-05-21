#include <iostream>
#include <stdlib.h>
#include <sys/time.h>

using namespace std;

/* double precision matrix multiplication
 * (For performance comparison with GPU matrix multiplication)   
 */
void matmul(int a1, int a2, double **a, int b2, double **b, double **c) {

	for (int i=0; i<a1; i++) {
		double sum[b2];
		for (int j=0; j<b2; j++)
			sum[j]=0.0;
		for (int k=0; k<a2; k++) {
			for (int j=0; j<b2; j++) {
				sum[j] += a[i][k]*b[k][j];
			}
		}
		for (int j=0; j<b2; j++)
			c[i][j] = sum[j];
	}
}

int main() {

	time_t sTime = time(NULL);
        timeval tt1, tt2;
        int ms;
        double fms;

	int a1=2560;
	int a2=2560;
	int b2=2560;

	double **a = (double**)malloc(a1*sizeof(double*));
	for (int i=0; i< a1; i++)
	   a[i] = (double*)malloc(a2*sizeof(double));

	double **b = (double**)malloc(a2*sizeof(double*));
	for (int i=0; i< a2; i++)
	   b[i] = (double*)malloc(b2*sizeof(double));

	for (int i=0;i<a1;i++)
		for (int j=0;j<a2;j++)
			a[i][j] = (double)(i+j);

	for (int i=0;i<a2;i++)
		for (int j=0;j<b2;j++)
			b[i][j] = (double)(i-j);

	double **c = (double**)malloc(a1*sizeof(double*));
	for (int i=0; i< a1; i++)
	   c[i] = (double*)malloc(b2*sizeof(double));

	gettimeofday( &tt1, NULL );

	matmul(a1, a2, a, b2, b, c);

	gettimeofday( &tt2, NULL );

	ms = (tt2.tv_sec - tt1.tv_sec);
        ms = ms * 1000000 + (tt2.tv_usec - tt1.tv_usec);
        fms = ((double)ms)/1000000.0;
        cout << "Comp time = " << fms << endl;

	double dNumOps = 2.0 * (double)a1 * (double)a2 * (double)b2;
        double gflops = 1.0e-9 * dNumOps/fms;
        cout << "GFlops = " << gflops << endl;

	cout << c[145][999] << " " << c[232][567] << endl;
}



