! First compile fortran_thunking.c:
! icc  -I/usr/local/cuda/include -I/usr/local/cuda/src -c fortran_thunking.c

! then complile this file:
! ifort -L/usr/local/cuda/lib64 cublas_thunking.f90 fortran_thunking.o -lcublas

program cublas

integer::n,lda,ldb,ldc,i,j
real, allocatable::a(:,:),b(:,:),c(:,:)
real*8::t1,t2,tdiff

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

 call cpu_time(t1)
 call cublas_SGEMM('N','N',n,n,n,1.0,a,lda,b,ldb,0.0,c,ldc)
 call cpu_time(t2)

 print *, 'value check = ', c(1,146)

deallocate(a,b,c)
flops = 2.0 * real(n) * real(n) * real(n)
tdiff = t2-t1
write(*,'(i5,f8.5,i5)') n, tdiff, int(1.0e-9* flops/tdiff)

end
