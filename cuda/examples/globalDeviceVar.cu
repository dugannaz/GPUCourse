#include <iostream>
#include <cuda.h>

using namespace std;

// device variable
__device__ float d_test[2][2];

__global__ void kernel1() { 

	d_test[1][1] = 1.0; 
}


int main() {

  float h_test = 0.0;
  cudaMemset(&d_test,0,4*sizeof(float));

  // invoke kernel
  kernel1 <<<1,1>>> ();

  // Use cudaMemcpyFromSymbol instead of cudaMemcpy
  cudaMemcpyFromSymbol(&h_test, d_test, sizeof(float), 3*sizeof(float), cudaMemcpyDeviceToHost);

  cout << h_test << endl;  
}

