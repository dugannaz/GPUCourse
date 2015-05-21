#include <iostream>
#include <stdlib.h>
#include <sys/time.h>

#define TILE_DIM 64

using namespace std;

void matmul_shared(int a1, int a2, double **a, int b2, double **b, double **c) {

	#pragma omp parallel 
        {
	// (*) create 2D cache arrays of size (TILE_DIM x TILE_DIM) for matrices (a)(b)(c)
	double aTile[TILE_DIM][TILE_DIM]; 
	double bTile[TILE_DIM][TILE_DIM];
	double cTile[TILE_DIM][TILE_DIM];

	#pragma omp for
	for (int i=0; i<a1/TILE_DIM; i++) {
		for (int j=0; j<b2/TILE_DIM; j++) {

			for (int it=0; it<TILE_DIM; it++) 
			for (int jt=0; jt<TILE_DIM; jt++)
				cTile[it][jt] = 0.0;

			// This loop is necessary since (aw) and (bh) are multiple of TILE_DIM 
			for (int ia=0; ia<a2; ia+=TILE_DIM) {

				// copy block data of iteration (ia) to cache
				for (int it=0; it<TILE_DIM; it++) 
				for (int jt=0; jt<TILE_DIM; jt++) 
					aTile[it][jt] = a[i*TILE_DIM+it][jt + ia];

				for (int it=0; it<TILE_DIM; it++) 
				for (int jt=0; jt<TILE_DIM; jt++) 
					bTile[it][jt] = b[it+ia][j*TILE_DIM+jt];			

				// multiplication
				for (int it=0; it<TILE_DIM; it++)  
				for (int k = 0; k < TILE_DIM; k++)
				for (int jt=0; jt<TILE_DIM; jt++) 
					cTile[it][jt] += aTile[it][k]* bTile[k][jt];
		
			}

			// write back to memory
			for (int it=0; it<TILE_DIM; it++) 
			for (int jt=0; jt<TILE_DIM; jt++) 
				c[i*TILE_DIM+it][j*TILE_DIM+jt] = cTile[it][jt];
		}
	}

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

	matmul_shared(a1, a2, a, b2, b, c);
	

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



