program compare_performance
    use iso_fortran_env
    use, intrinsic :: iso_c_binding
    implicit none
    integer, parameter :: N1 = 5000, N2 = 10000, iterations = 10000
    integer :: i
    real(kind=c_float), dimension(:,:), allocatable :: matrix
    real(kind=c_float), dimension(:), allocatable :: vector
    real(kind=c_float), dimension(:), allocatable :: result
    real :: start_time, end_time

    !call test(N1)
      
    !call test(N2)
    
    
   
    
    call test_matmul


    read(*,*)
    
contains

    subroutine test(N)
        integer, intent(in) :: N
        allocate(matrix(N, N), vector(N), result(N))

        ! Initialize matrix and vector with random values
        call random_number(matrix)
        call random_number(vector)

        print *, "Comparing performance for dimension N = ", N

        ! CPU benchmark
        call cpu_time(start_time)
        !$omp parallel do
        do i = 0, iterations
            result = matmul(matrix, vector)
        end do
        !$omp end parallel do
        call cpu_time(end_time)
        print *, "CPU Time: ", end_time - start_time, " seconds"
        print *, "Matmul Result: ", result
        
        deallocate(matrix, vector, result)
    end subroutine test
    
    
    subroutine test_matmul  
        
        integer, parameter :: N = 5000
        integer, parameter :: M =  5000
        integer, parameter ::  TIMES = 10000
        integer :: i, j, k, l, Nl, l1, l2, p    
        real :: A(N, M), x(N), b(M)

    
        real :: t1, t2
    
         do i=1, N
            do j=1, M 
               A(i,j)   = (i-1) * (j+1)  / real( N * M) 
            end do 
            x(i) = (i-1) * (i-1)  / real( N * M ) 
         end do 
    
    call CPU_TIME(t1)
    do k=0, TIMES  
            b = matmul(A, x) 
           ! write(*,*) k
    end do 
    call CPU_TIME(t2)
    write(*,*) " CPU Time = ", t2-t1 
    write(*,*) "b_1 = ", b(0) 
    write(*,*) "b_M = ", b(M) 
    
    
    call CPU_TIME(t1)
    do k=0, TIMES  
        
        b = 0 
        do j=1, M
          do i=1, N
              b(i) = b(i) + A(i,j) * x(j)
          end do 
        end do 
        
     !   write(*,*) k
    end do 
    
        call CPU_TIME(t2)
        write(*,*) " CPU Time= ", t2-t1 
        write(*,*) "b_1 = ", b(0) 
        write(*,*) "b_M = ", b(M) 
        
    end subroutine

end program compare_performance

    
    
    