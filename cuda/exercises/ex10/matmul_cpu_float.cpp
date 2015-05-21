#include <iostream>
#include <stdlib.h>
#include <sys/time.h>

using namespace std;

/* single precision matrix multiplication
 * (For performance comparison with GPU matrix multiplication)   
 */
void matmul(int a1, int a2, float **a, int b2, float **b, float **c) {

	for (int i=0; i<a1; i++) {
		float sum[b2];
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
        float fms;

	int a1=2560;
	int a2=2560;
	int b2=2560;

	float **a = (float**)malloc(a1*sizeof(float*));
	for (int i=0; i< a1; i++)
	   a[i] = (float*)malloc(a2*sizeof(float));

	float **b = (float**)malloc(a2*sizeof(float*));
	for (int i=0; i< a2; i++)
	   b[i] = (float*)malloc(b2*sizeof(float));

	for (int i=0;i<a1;i++)
		for (int j=0;j<a2;j++)
			a[i][j] = (float)(i+j);

	for (int i=0;i<a2;i++)
		for (int j=0;j<b2;j++)
			b[i][j] = (float)(i-j);

	float **c = (float**)malloc(a1*sizeof(float*));
	for (int i=0; i< a1; i++)
	   c[i] = (float*)malloc(b2*sizeof(float));

	gettimeofday( &tt1, NULL );

	matmul(a1, a2, a, b2, b, c);

	gettimeofday( &tt2, NULL );

	ms = (tt2.tv_sec - tt1.tv_sec);
        ms = ms * 1000000 + (tt2.tv_usec - tt1.tv_usec);
        fms = ((float)ms)/1000000.0;
        cout << "Comp time = " << fms << endl;

	float dNumOps = 2.0 * (float)a1 * (float)a2 * (float)b2;
        float gflops = 1.0e-9 * dNumOps/fms;
        cout << "GFlops = " << gflops << endl;

	cout << c[145][999] << " " << c[232][567] << endl;
}



