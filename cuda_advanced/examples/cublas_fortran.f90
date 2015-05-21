! First compile cublas_fortran.cpp
! nvcc -arch sm_20 -c cublas_fortran.cpp -lcublas

! then compile this file
! ifort cublas_fortran.f90 cublas_fortran.o -lcudart -lcublas -L/usr/local/cuda/lib64/

program cublas

integer::n,lda,ldb,ldc, stat, zero_int
real*8, allocatable::a(:,:),b(:,:),c(:,:)
real*8::t1,t2,tdiff, zero, one

zero_int = 0
zero = 0.0
one = 1.0

n = 2560

lda = n
ldb = n
ldc = n

allocate(a(lda,n), b(ldb,n), c(ldc,n))

 do i=1,n
  do j=1,lda
    a(i,j) = real(i+j)
    b(i,j) = real(i-j)
  enddo
 enddo

 !call random_number(a)
 !call random_number(b)

 call cublasDGEMM(zero_int, zero_int, n, n, n, one, a, lda, b, ldb, zero, c, ldc)

 print *, 'value check = ', c(1,146)

deallocate(a,b,c)

end
