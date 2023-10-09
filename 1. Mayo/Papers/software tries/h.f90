

! sAXPY example using Do Concurrent construct in Fortran
! Build with
!   nvfortran -stdpar -Minfo -fast saxpy.f90
! Build with to target Multicore
!   nvfortran -stdpar=multicore -Minfo=accel -fast saxpy.f90
!

module sm

contains

 subroutine saxpy_concurrent(x, y, n, a)
        real :: x(:) , y(:)
        real :: a
        integer :: n, i  
		
        do concurrent (i = 1: n)
		
          y(i) = a * x(i) + y(i) 
		  
        enddo  
		
 end subroutine 

subroutine saxpy_do(x, y, a) 
        real, intent(in) :: x(:), a 
        real, intent(inout) :: y(:) 
        
        integer ::  i 
		
        do i = 1, size(x) 
		
          y(i) = a * x(i) + y(i)
		  
        enddo 
		
 end subroutine  
	
end module

program main

    use sm
	  
    real,allocatable :: x(:), y1(:), y2(:)
    real :: a = 2.0 
    integer :: n, i, j, err = 0
    integer :: c0, c1, c2, cpar, cseq
    real :: t1, t2, t3  
    n = 100000000
    m= 1000 
    allocate(  x(n), y1(n), y2(n) )
    
    call random_number( x )
    call random_number( y1 )
    call random_number( y2 )   
 
    
    call system_clock( count=c0 )
    call CPU_TIME(t1) 
    do j=1, m 
      call saxpy_do(x, y1, a)
    end do 
    call system_clock( count=c1 )
    call CPU_TIME(t2)
    
    do j=1, m 
       call saxpy_concurrent(x, y2, n, a)
    end do 
    call CPU_TIME(t3)
    
    call system_clock( count=c2 )
    cseq = c1 - c0
    cpar = c2 - c1  

    
    print *, cseq, ' microseconds sequential'
    print *, t2-t1, 'seconds sequential'
    print *, cpar, ' microsecondswith stdpar'
    print *, t3-t2, 'seconds  parallel'
    
      write(*,*) sum(y1/n) 
      write(*,*) sum(y2)/n 
   
end program