#include <iostream>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>
// (*) include cufft library here

/* CUFFT library
 * - 2D FFT transform using 1D cuffts
 * - compare with 2D cufft 
 */

#define NX 2048
#define BATCH NX
#define TILE_DIM  16

using namespace std;

// (*) modify function definition for cufftDoubleComplex data type
__global__ void transposeNoBankConflicts(float *idata, float *odata, int width, int height)
{
  __shared__ float tile[TILE_DIM][TILE_DIM+1];

  int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
  int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
  int index_in = xIndex + (yIndex)*width;

  xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
  yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
  int index_out = xIndex + (yIndex)*height;
  
  // (*) modify here to transpose also the complex part of the matrix

      tile[threadIdx.y][threadIdx.x] = idata[index_in];

    __syncthreads();
   
      odata[index_out] = tile[threadIdx.x][threadIdx.y];   
  
}

int main(int argc, char *argv[]) {

        struct timeval tt1, tt2;
        int ms;
        float fms;

	// (*) create cufft plan
        
        
  	// (*) allocate cufftDoubleComplex type host memory 
        cufftDoubleComplex *data;
	data = (cufftDoubleComplex*)malloc(/* fill here */);

	// data initialization
         for(int j=0 ; j < BATCH ; j++)
           for(int k=0 ; k < NX ; k++) {
                data[k + j*NX].x = sin(double(j)+double(k));
                data[k + j*NX].y = cos(double(j)+double(k));
           }

	// check initial value of a data element 
        cout << "initial value = " << data[43].x << " + " 
	     << data[43].y << "i" << endl;

	// (*) allocate cufftDoubleComplex type device memory
	cufftDoubleComplex *devPtr;
        cudaMalloc((void**)&devPtr, /* fill here */);

	// copy data to device memory
        cudaMemcpy(devPtr, data, sizeof(cufftDoubleComplex)*NX*BATCH, cudaMemcpyHostToDevice);

	// runtime configuration parameters for transposition
        dim3 grid(NX/TILE_DIM,NX/TILE_DIM,1);
        dim3 threads(TILE_DIM,TILE_DIM,1);

        cudaThreadSynchronize();
        gettimeofday( &tt1, NULL );	

        // (*) run fft and transpostion kernels for both dimensions
        

        cudaThreadSynchronize();
        gettimeofday( &tt2, NULL );

        // (*) make inverse transform 
        

	// transfer result back from device
        cudaMemcpy(data, devPtr, sizeof(cufftDoubleComplex)*NX*BATCH, cudaMemcpyDeviceToHost);

        // (*) destroy cufft plan
        

	// free device memory
        cudaFree(devPtr);

	// check initial value of the same data element. Initial and final values should match
	// after a forward and inverse transform. 
        cout << "final value   = " << data[43].x/double(NX*NX) << " + " 
	     << data[43].y/double(NX*NX) << "i" << endl;

	// free host memory
	free(data);

	// timing
        ms = (tt2.tv_sec - tt1.tv_sec);
        ms = ms * 1000000 + (tt2.tv_usec - tt1.tv_usec);
        fms = ((double)ms)/1000000.f;
        cout << "Computation time = " << fms << " seconds" << endl;

}

