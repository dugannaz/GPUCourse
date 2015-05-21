PROGRAM call_cuda

     	implicit none

      	integer, parameter  :: n = 2048
	integer, parameter  :: nBlocks = 32
	integer, parameter  :: nThreads = 64

        ! call cuda interface function
      	call kernel_wrapper(n, nBlocks, nThreads)
      
END PROGRAM call_cuda
