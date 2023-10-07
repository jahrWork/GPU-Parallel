    program fortran_mkl
    
    use omp_lib
    use iso_fortran_env
    implicit none
    
    real, allocatable :: A(:,:), B(:,:), C(:,:)
    real :: power, results(163), dimensions(163)
    integer :: j
    integer (kind=8) :: i
    integer :: rate, t0, t1
    integer (kind=int64) :: N_ops, N
    integer (kind=8) :: TIMES
    
    
    N_ops = 2 * 10000_8**3
    write(*,*) N_ops
    
    ! Warm up section
    N = 5000; TIMES = 2
    allocate(A(N, N), B(N, N), C(N, N))
    call random_number(A)
    call random_number(B)
    C = 0
    do i = 1, TIMES
        call sgemm("N", "N", N, N, N, 1e0, A, N, B, N, 0e0, C, N)
    end do
    deallocate(A, B, C)
    
    j = 0
    do N = 50, 2500, 25
    
        j = j + 1
        
        TIMES = 10000_8**3 / N**3
        write(*,*) "N = ", N, "TIMES = ", TIMES
        dimensions(j) = N
        
        allocate(A(N, N), B(N, N), C(N, N))
        
        call random_number(A)
        call random_number(B)
        
        C = 0
        call test_vectorized_and_parallelized_series(N)
        call system_clock(t0)
        call system_clock(count_rate=rate)
        
        !!$omp parallel do private(i)
        do i = 1, TIMES
            call sgemm("N", "N", N, N, N, 1e0, A, N, B, N, 0e0, C, N)
        end do
        !!$omp end parallel do
        
        call system_clock(t1)
        write(*,*) "Execution time:", (t1 - t0)/real(rate), "seconds"
        results(j) = (t1 - t0)/real(rate)
        
        deallocate(A, B, C)
        
    end do
    
    do TIMES = 63, 1, -1
        
        j = j + 1
        
        power = 1./3
        N = nint((real(10000_8**3) / TIMES)**power)
        dimensions(j) = N
        
        write(*,*) "N = ", N, "TIMES = ", TIMES
        
        allocate(A(N, N), B(N, N), C(N, N))
        
        call random_number(A)
        call random_number(B)
        
        C = 0
        call test_vectorized_and_parallelized_series(N)
        call system_clock(t0)
        call system_clock(count_rate=rate)
        
        !!$omp parallel do private(i)
        do i = 1, TIMES
            call sgemm("N", "N", N, N, N, 1e0, A, N, B, N, 0e0, C, N)
        end do
        !!$omp end parallel do
        
        call system_clock(t1)
        write(*,*) "Execution time:", (t1 - t0)/real(rate), "seconds"
        results(j) = (t1 - t0)/real(rate)
        
        deallocate(A, B, C)
        
    end do
    
    write(*,*) "[", dimensions, "]"
    write(*,*) "[", results, "]"
    
    write(*,*) "Write anything and press Enter to finish"
    read(*,*)
    
    contains
    
    subroutine test_vectorized_and_parallelized_series(N)
    
    integer(kind=8) , intent(in) ::  N
    integer :: N_threads_max, i, imax, Nt  
     
    N_threads_max = omp_get_max_threads()
    write(*,*) "Max number of threads =", N_threads_max
    
    end subroutine 
    
    
    end program fortran_mkl