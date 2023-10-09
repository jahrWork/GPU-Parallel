
 
 
!**************************************************    
! It performs sum from n=1 to infinity of 1/n**2 
! with a coarray associated to different images 
!     
!  Author: Nov 2021, juanantonio.hernandez@upm.es 
!**************************************************    
module Series_with_coarrays

contains 
subroutine example_Series_with_coarrays

implicit none  

! this image and number of images 
  integer :: image, j, Ni  

! S is a coarray of dimension  determined at runtime
  real (kind=8), save  :: S[*]  
  
! SN is the total sum of every image   
  real (kind=8) :: SN
  real (kind=8) :: t0, tf, rate, PI = 4*atan(1.0)
  
! N is the total number of terms 
! Nt is the total number of terms of each image   
  integer(kind=8) :: i,  N , Nt 

   call cpu_time(t0) 
   image = this_image()
   Ni = num_images()
   
!  number of terms to be added    
   N = 2.**38
!  number of terms for each image    
   Nt = N / Ni
 
!  Each image performs a backwards sum of the total sum  
   S = 0 
   do i = image * Nt, 1 + (image-1)*Nt,  -1
      S = S + 1 / real(i)**2
   end do
   
!  partial result of each image    
   write(*,*) " image =", image, " S = ", S  
   
 ! Once image 1 finishes, it sums the contribution of each image
   if (image .eq. 1) then
      
     write(*,*) "Sum contributions of different images  " 
     SN = 0   
     do j = Ni, 1, -1
       SN = SN + S[j] 
     end do
     
     call cpu_time(tf)
     write (*,*) "CPU time:", tf-t0
     write (*,*) "Number of images:", Ni
     write (*,*) "Number of terms =", N
     write (*,*) "Calculated  SN=", SN
     write (*,*) "Error=         ", PI**2/6 - SN
     write (*,*) "Error bound =  ", 1./N 
     
   end if

end subroutine    
end module   
    

    
    
    
    
    
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
    
   