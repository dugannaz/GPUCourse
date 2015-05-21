#include <iostream>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <cufft.h>

#define NX 2048

using namespace std;

int main(int argc, char *argv[]) {

        struct timeval tt1, tt2;
        int ms;
        float fms;

	// create cufft plan
        cufftHandle plan;
        cufftPlan2d(&plan, NX,NX, CUFFT_Z2Z);

  	// allocate cufftDoubleComplex type host memory 
        cufftDoubleComplex *data;
	data = (cufftDoubleComplex*)malloc(NX*NX * sizeof(cufftDoubleComplex));

	// data initialization
         for(int j=0 ; j < NX ; j++)
           for(int k=0 ; k < NX ; k++) {
                data[k + j*NX].x = sin(double(j)+double(k));
                data[k + j*NX].y = cos(double(j)+double(k));
           }

	// check initial value of a data element 
        cout << "initial value = " << data[43].x << " + " 
		<< data[43].y << "i" << endl;

	// allocate cufftDoubleComplex type device memory
	cufftDoubleComplex *devPtr;
        cudaMalloc((void**)&devPtr, sizeof(cufftDoubleComplex)*NX*NX);

	// copy data to device memory
        cudaMemcpy(devPtr, data, sizeof(cufftDoubleComplex)*NX*NX, cudaMemcpyHostToDevice);

        cudaThreadSynchronize();
        gettimeofday( &tt1, NULL );

        // run fft 
      
        cufftExecZ2Z(plan, devPtr, devPtr, CUFFT_FORWARD);	

        cudaThreadSynchronize();
        gettimeofday( &tt2, NULL );

        // make inverse transform 
        
        cufftExecZ2Z(plan, devPtr, devPtr, CUFFT_INVERSE);	

	// transfer result back from device
        cudaMemcpy(data, devPtr, sizeof(cufftDoubleComplex)*NX*NX, cudaMemcpyDeviceToHost);

        // destroy cufft plan
        cufftDestroy(plan);

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

