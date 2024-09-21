 module parallel_matmul 
    
   implicit none 
   integer, parameter :: N = 5000, M= 5000
   real, save :: A(N, M), x(M), b(N), A1(N/4,N), A2(N/4,N), A3(N/4,N), A4(N/4,N)
   !integer, parameter :: M = 5000
  ! real, save, allocatable :: A(:, :), x(:), b(:)
   
   
    contains 
    
    
    
    
!subroutine test_matmul_single_core(N_operations)
!    integer(kind=8), intent(out) :: N_operations
!    integer :: N, M, block_size
!
!    real, allocatable :: A(:,:), x(:), b(:)
!    real, allocatable :: block(:,:)
!    integer :: k, i, upper_idx
!    integer :: it1
!    integer :: TIMES = 1000
!    integer, parameter :: cache_size = 32 * 1024 * 1024 ! 32MB
!    real :: bytes_per_element
!
!    ! Define N and M here (as an example)
!    N = 5000
!    M = 5000
!
!    bytes_per_element = 4.0 ! 4 bytes for single precision
!    block_size = int((cache_size - M * bytes_per_element) / (M * bytes_per_element + bytes_per_element))
!
!    N_operations = 2 * N * M * real(TIMES)
!
!    allocate(A(N, M), x(M), b(N))
!
!    call Initialization(N, M, A, x)
!
!    call system_clock(it1)
!    do k = 1, TIMES
!        do i = 1, N, block_size
!            upper_idx = min(i + block_size - 1, N)
!            block = A(i : upper_idx, :)
!            b(i : upper_idx) = matmul(block, x)
!        end do
!    end do
!    call final_cpu_time1("matmul_blocked", it1, b)
!
!    deallocate(A, x, b)
!
!end subroutine test_matmul_single_core

!subroutine test_matmul_single_core(N_operations)
!    implicit none
!    integer :: N=5000, M=5000
!    integer(kind=8), intent(out) :: N_operations
!    real, allocatable, target :: A(:,:), x(:), b(:)
!    real, pointer :: A_blocks(:,:)
!    integer :: k, i
!    integer :: it1
!    integer, parameter :: TIMES = 1000
!    integer, parameter :: NUM_BLOCKS = 4
!    integer :: blockSize
!
!    ! Allocate memory
!    allocate(A(N, M), x(M), b(N))
!
!    N_operations = 2 * N * M * real(TIMES)
!
!    ! Assume Initialization is a custom subroutine you have defined elsewhere
!    call Initialization(N, M, A, x)
!
!    blockSize = N / NUM_BLOCKS
!
!    call system_clock(it1)
!    do k = 1, TIMES
!        do i = 0, NUM_BLOCKS - 1
!            A_blocks => A(i * blockSize + 1 : (i + 1) * blockSize, :)
!            b(i * blockSize + 1 : (i + 1) * blockSize) = matmul(A_blocks, x)
!        end do
!    end do
!    call final_cpu_time1("matmul_blocked", it1, b)  ! Make sure final_cpu_time1 is defined properly elsewhere.
!
!    ! Deallocate memory
!    deallocate(A, x, b)
!end subroutine


    
subroutine test_matmul_single_core( N_operations ) 
 integer(kind=8), intent(out) :: N_operations 
        
 !real, allocatable :: A(:, :), x(:), b(:)
 integer :: k
 integer :: it1
  
    integer :: TIMES = 1000
   ! allocate( A(N,N), x(N), b(N) ) 
   
    N_operations = 2 * N * M * real(TIMES)
    
    
    call Initialization(N, M, A, x) 
 
     A1 = A(1:N/4,:)
     A2 = A(N/4+1:2*N/4,:)
     A3 = A(2*N/4+1:3*N/4,:)
     A4 = A(3*N/4+1:N,:)
   
   
    call system_clock(it1)
    do k=0, TIMES  
            
            b(1:N/4) = matmul(A1,x)     
            b(N/4+1:2*N/4) = matmul(A2,x)!(A(N/4+1:2*N/4,:), x)
            b(2*N/4+1:3*N/4) = matmul(A3,x)!(A(2*N/4+1:3*N/4,:), x)
            b(3*N/4+1:N) = matmul(A4,x)!(A(3*N/4+1:N,:), x)
            
            
            !write(*,*) k
    end do 
    call final_cpu_time1("matmul_blocked", it1, b) 
    
   read(*,*)
   !
   !   call system_clock(it1)
   ! do k=0, TIMES  
   !     b = my_matmul(A, x) 
   ! !   write(*,*) k
   ! end do 
   !call final_cpu_time1("my_matmul", it1, b) 
   !
   ! read(*,*)
   
   end subroutine
        
        
        
 
    
!   
!   
!   !    call system_clock(it1)
!   ! do k=0, TIMES  
!   !     b = my_matmul_blocked_16(A, x) 
!   ! !   write(*,*) k
!   ! end do 
!   !call final_cpu_time1("my_matmul_blocked_16", it1, b) 
!   !
!   !
!   !
!   !   call system_clock(it1)
!   ! do k=0, TIMES  
!   !     b = my_matmul_blocked_32(A, x) 
!   ! !   write(*,*) k
!   ! end do 
!   !call final_cpu_time1("my_matmul_blocked_32", it1, b) 
!   !
!   !
!   !   call system_clock(it1)
!   ! do k=0, TIMES  
!   !     b = my_matmul_parallel(A, x) 
!   ! !   write(*,*) k
!   ! end do 
!   !call final_cpu_time1("my_matmul_parallel", it1, b) 
!   !
!   !call system_clock(it1)
!   ! do k=0, TIMES  
!   !     b = my_matmul_blocked_parallel_16(A, x) 
!   ! !   write(*,*) k
!   ! end do 
!   !call final_cpu_time1("my_matmul_blocked_parallel_16", it1, b) 
!   !
!   !
!   !call system_clock(it1)
!   ! do k=0, TIMES  
!   !     b = my_matmul_blocked_parallel_32(A, x) 
!   ! !   write(*,*) k
!   ! end do 
!   !call final_cpu_time1("my_matmul_blocked_parallel_32", it1, b) 
!   !
!   !call system_clock(it1)
!   ! do k=0, TIMES  
!   !     b = my_matmul_blocked_parallel_reduction_16(A, x) 
!   ! !   write(*,*) k
!   ! end do 
!   !call final_cpu_time1("my_matmul_blocked_parallel_reduction_16", it1, b)
!   !
!   !call system_clock(it1)
!   ! do k=0, TIMES  
!   !     b = my_matmul_blocked_parallel_reduction_32(A, x) 
!   ! !   write(*,*) k
!   ! end do 
!   !call final_cpu_time1("my_matmul_blocked_parallel_reduction_32", it1, b) 
!   !
!   
!   
!   write(*,*) " Press enter" 
!   read(*,*)
!
!  
!        
!end subroutine
!
!subroutine  Initialization(N, M, A, x)  
!    integer, intent(in) :: N, M 
!    real,intent(out) ::  A(:,:), x(:) 
!  
! 
!    integer :: i, j 
!  
!    do i=1, N
!            do j=1, M 
!               A(i,j)   = (i-1) * (j+1)  / real( N * M) 
!            end do 
!            x(i) = (i-1) * (i-1)  / real( N * M ) 
!    end do  
!
!end subroutine 
!
!function my_matmul(A, x) result(b) 
!   real, intent(in) :: A(:,:), x(:) !!Usar explicit A(n,n)
!   real :: b(size(A, dim=1))
!   
!   
!   integer N, M, i, j  
!   
!   N = size(A, dim=1) 
!   M = size(A, dim=2) 
!           
!   b = 0 
!   do j=1, M
!      do i=1, N
!              b(i) = b(i) + A(i,j) * x(j)
!      end do 
!   end do 
!
!end function 
!
!
!function my_matmul_blocked_16(A, x) result(b)
!    
!    real, intent(in) :: A(:,:), x(:)
!    real :: b(size(A, dim=1))
!    integer :: N, M, i, j, jj, ii, blockSize
!    integer, parameter :: cacheSize = 16 * 1024 * 1024  ! 32MB
!    
!    N = size(A, dim=1)
!    M = size(A, dim=2)
!    
!    blockSize = int(sqrt(real(cacheSize) / 4.0))  ! 4 bytes por elemento de tipo real
!    
!    b = 0
!    
!    do jj = 1, M, blockSize
!        do ii = 1, N, blockSize
!            do j = jj, min(jj + blockSize - 1, M)
!                do i = ii, min(ii + blockSize - 1, N)
!                    b(i) = b(i) + A(i, j) * x(j)
!                end do
!            end do
!        end do
!    end do
!
!end function
!
!
!function my_matmul_blocked_32(A, x) result(b)
!    
!    real, intent(in) :: A(:,:), x(:)
!    real :: b(size(A, dim=1))
!    integer :: N, M, i, j, jj, ii, blockSize
!    integer, parameter :: cacheSize = 32 * 1024 * 1024  ! 32MB
!    
!    N = size(A, dim=1)
!    M = size(A, dim=2)
!    
!    blockSize = int(sqrt(real(cacheSize) / 4.0))  ! 4 bytes por elemento de tipo real
!    
!    b = 0
!    
!    do jj = 1, M, blockSize
!        do ii = 1, N, blockSize
!            do j = jj, min(jj + blockSize - 1, M)
!                do i = ii, min(ii + blockSize - 1, N)
!                    b(i) = b(i) + A(i, j) * x(j)
!                end do
!            end do
!        end do
!    end do
!
!end function
!
!
!function my_matmul_parallel(A, x) result(b)
!    use omp_lib, only: omp_set_num_threads, omp_get_max_threads
!    implicit none
!    real, intent(in) :: A(:,:), x(:)
!    real :: b(size(A, dim=1))
!    integer :: N, M, i, j
!    
!    N = size(A, dim=1)
!    M = size(A, dim=2)
!    
!    b = 0
!    
!    ! Establecer el número de hilos al máximo disponible
!    !call omp_set_num_threads(omp_get_max_threads())
!
!    !$omp parallel do private(j)
!    do i = 1, N
!        do j = 1, M
!            b(i) = b(i) + A(i, j) * x(j)
!        end do
!    end do
!    !$omp end parallel do
!
!end function my_matmul_parallel
!
!function my_matmul_blocked_parallel_16(A, x) result(b) ! SU RESULTADO ES INCORRECTO, NECESARIO USAR "REDUCTION"
!    
!    use omp_lib
!    
!    real, intent(in) :: A(:,:), x(:)
!    real :: b(size(A, dim=1))
!    integer :: N, M, i, j, jj, ii, blockSize
!    integer, parameter :: cacheSize = 16 * 1024 * 1024  ! 32MB
!    
!    N = size(A, dim=1)
!    M = size(A, dim=2)
!    
!    blockSize = int(sqrt(real(cacheSize) / 4.0))  ! 4 bytes por elemento de tipo real
!    
!    b = 0
!    
!    !$omp parallel private(i, j)
!    do jj = 1, M, blockSize
!        do ii = 1, N, blockSize
!            !$omp do
!            do j = jj, min(jj + blockSize - 1, M)
!                do i = ii, min(ii + blockSize - 1, N)
!                    b(i) = b(i) + A(i, j) * x(j)
!                end do
!            end do
!            !$omp end do
!        end do
!    end do
!    !$omp end parallel
!
!end function
!
!
!function my_matmul_blocked_parallel_32(A, x) result(b) ! SU RESULTADO ES INCORRECTO, NECESARIO USAR "REDUCTION"
!    
!    use omp_lib
!    
!    real, intent(in) :: A(:,:), x(:)
!    real :: b(size(A, dim=1))
!    integer :: N, M, i, j, jj, ii, blockSize
!    integer, parameter :: cacheSize = 32 * 1024 * 1024  ! 32MB
!    
!    N = size(A, dim=1)
!    M = size(A, dim=2)
!    
!    blockSize = int(sqrt(real(cacheSize) / 4.0))  ! 4 bytes por elemento de tipo real
!    
!    b = 0
!    
!    !$omp parallel private(i, j)
!    do jj = 1, M, blockSize
!        do ii = 1, N, blockSize
!            !$omp do
!            do j = jj, min(jj + blockSize - 1, M)
!                do i = ii, min(ii + blockSize - 1, N)
!                    b(i) = b(i) + A(i, j) * x(j)
!                end do
!            end do
!            !$omp end do
!        end do
!    end do
!    !$omp end parallel
!
!end function
!
!
!function my_matmul_blocked_parallel_reduction_16(A, x) result(b)
!    
!    use omp_lib
!    
!    real, intent(in) :: A(:,:), x(:)
!    real :: b(size(A, dim=1))
!    integer :: N, M, i, j, jj, ii, blockSize
!    integer, parameter :: cacheSize = 16 * 1024 * 1024  ! 32MB
!    
!    N = size(A, dim=1)
!    M = size(A, dim=2)
!    
!    blockSize = int(sqrt(real(cacheSize) / 4.0))  ! 4 bytes per real element
!    
!    b = 0
!    
!    !$omp parallel private(jj, ii, i, j) 
!    do jj = 1, M, blockSize
!        do ii = 1, N, blockSize
!            !$omp do reduction(+:b)
!            do j = jj, min(jj + blockSize - 1, M)
!                do i = ii, min(ii + blockSize - 1, N)
!                    b(i) = b(i) + A(i, j) * x(j)
!                end do
!            end do
!            !$omp end do
!        end do
!    end do
!    !$omp end parallel
!
!end function
!
!function my_matmul_blocked_parallel_reduction_32(A, x) result(b)
!    
!    use omp_lib
!    
!    real, intent(in) :: A(:,:), x(:)
!    real :: b(size(A, dim=1))
!    integer :: N, M, i, j, jj, ii, blockSize
!    integer, parameter :: cacheSize = 32 * 1024 * 1024  ! 32MB
!    
!    N = size(A, dim=1)
!    M = size(A, dim=2)
!    
!    blockSize = int(sqrt(real(cacheSize) / 4.0))  ! 4 bytes per real element
!    
!    b = 0
!    
!    !$omp parallel private(jj, ii, i, j) 
!    do jj = 1, M, blockSize
!        do ii = 1, N, blockSize
!            !$omp do reduction(+:b)
!            do j = jj, min(jj + blockSize - 1, M)
!                do i = ii, min(ii + blockSize - 1, N)
!                    b(i) = b(i) + A(i, j) * x(j)
!                end do
!            end do
!            !$omp end do
!        end do
!    end do
!    !$omp end parallel
!
!end function




subroutine  Initialization(N, M, A, x)  
    integer, intent(in) :: N, M 
    real,intent(out) ::  A(:,:), x(:) 
  
 
    integer :: i, j 
  
    do i=1, N
            do j=1, M 
               A(i,j)   = (i-1) * (j+1)  / real( N * M) 
            end do 
            x(i) = (i-1) * (i-1)  / real( N * M ) 
    end do  

end subroutine 




subroutine  final_cpu_time1(test, it1, b)
       character(len=*), intent(in) :: test 
       integer, intent(in) :: it1
       real, intent(in) :: b(:) 
  
    integer :: it2, rate  
  
   
    call system_clock(count_rate=rate)
    call system_clock(it2)
    
    write(*,'(A18, A)') "----- Testing... ", test 

    write(*,*) " CPU_time = ", real(it2-it1)/rate 
    write(*,*) "b_1 = ", b(1) 
    write(*,*) "b_M = ", b( size(b) ) 
    write(*,*) 
    
end subroutine     


end module 