 module matmul_benchmark 
    
   use CPU_specifications  
   implicit none 
   integer, parameter :: N = 1000, M= 1000
   real, save :: A(N, M), x(M), b(N)
   private 
   public :: test_matmul
   
   interface 
     function interface_matmul(A, x) result(b) 
      real, intent(in) :: A(:,:), x(:) 
      real :: b(size(A, dim=1))
     end function
   end interface 
   
    interface 
     function interface_matmul_explicit(N, A, x) result(b) 
      integer, intent(in) :: N
      real, intent(in) :: A(N,N), x(N) 
      real :: b(N)
     end function
   end interface 
   
   
    contains   
    
subroutine test_matmul 

   integer :: option 
   
do 
     
     
     write(*,*) " select an option " 
     write(*,*) " 0. exit/quit  "
     write(*,'(A100)') " 1. BEST performance with intrisic matmul single core, static memory                            " 
     write(*,'(A100)') " 2. POOR performance: my_matmul (row by column). assumed shape, single core, static memory      " 
     write(*,'(A100)') " 3. MEDIUM performance: my_matmul (column operations), assumed shape, single core, static memory" 
     write(*,'(A100)') " 4. BEST performance: my_matmul (column operations)), explicit shape, single core, static memory" 
     write(*,'(A100)') " 5. Intrinsic matmul single core, allocation with different dimensions                          " 
     write(*,'(A100)') " 6. BEST My matmul single core, allocation with different dimensions                            "
     write(*,'(A100)') " 7. Intrinsic matmul single core, different dimensions (plots)                                  "
     write(*,'(A100)') " 8. BEST My matmul single core, different dimensions (plots)                                    "
     write(*,'(A100)') " 9. Block my_matmul single core, different dimensions (plots)                                   "
     write(*,'(A100)') " 10. last                                   "
     
     
     
     read(*,*) option   
     
    select case(option)
    
    case(0) 
        exit
          
    case(1) 
        call matmul_single_core_static_memory
          
    case(2)
        call matmul_single_core_static_memory(poor_my_matmul) 
       
    case(3)
        call matmul_single_core_static_memory(medium_my_matmul) 
          
    case(4)
        call explicit_shape_matmul_single_core_static_memory(best_my_matmul) 
    
    case(5)  
        call explicit_shape_single_core_allocation_different_dimension 
         
    case(6)
        call explicit_shape_single_core_allocation_different_dimension(best_my_matmul)
         
    case(7)
        call explicit_shape_different_dimension_plots
        
    case(8)
        call explicit_shape_different_dimension_plots(best_my_matmul)
        
    case(9)    
         call block_matmul_test(best_my_matmul)
         call explicit_shape_different_dimension_plots(best_my_matmul)
         
    case(10)    
         call get_size_and_call_matmul(A, x, b)
        
        case default
        write(*,*) " option not implemented"
        
    end select 
     
end do
    
  
end subroutine 



subroutine matmul_single_core_static_memory(my_matmul) 
procedure (interface_matmul), optional :: my_matmul  
        
 
 integer :: k, t0
  
    integer :: TIMES = 25000
    integer (kind=8) N_op
     
    N_op = 2 * N * M * real(TIMES)
    
    write(*,*) "_______________________________________"
    write(*,*) " Ax = b matmul with N = ", N 
    write(*,*) " TIMES = ", TIMES
    call Initialization(N, M, A, x)  
   
    call system_clock(t0)
    do k=0, TIMES 
        if (present(my_matmul)) then 
               b = my_matmul(A, x) 
       else 
               b = matmul(A, x)
       end if 
    end do 
    call print_CPU_time(single_precision = .true., start_time = t0, N_operations = N_op, N_threads = 1) 

  
        
end subroutine

function poor_my_matmul(A, x) result(b) 
   real, intent(in) :: A(:,:), x(:) 
   real :: b(size(A, dim=1))
   
   
   integer N, M, i, j  
   
   N = size(A, dim=1) 
   M = size(A, dim=2) 
           
   b = 0 
   do i=1, N
      do j=1, M
              b(i) = b(i) + A(i,j) * x(j)
      end do 
   end do 

end function 

function medium_my_matmul(A, x) result(b) 
   real, intent(in) :: A(:,:), x(:) 
   real :: b(size(A, dim=1))
   
   
   integer N, M, i, j  
   
   N = size(A, dim=1) 
   M = size(A, dim=2) 
           
   b = 0 
   do j=1, M
      do i=1, N
              b(i) = b(i) + A(i,j) * x(j)
      end do 
   end do 

end function 

function best_my_matmul(N, A, x) result(b) 
   integer, intent(in) :: N 
   real, intent(in) :: A(N, N), x(N) 
   real :: b(N)
   
   integer  i, j 
           
   b = 0 
   do j=1, N
      do i=1, N
              b(i) = b(i) + A(i,j) * x(j)
      end do 
   end do 

end function 

subroutine get_size_and_call_matmul(A, x, b)

    real, intent(in) :: A(:,:), x(:)
    real, intent(out) :: b(size(x))
    integer :: N

    N = size(x)
    b = best_my_matmul(N, A, x)
end subroutine

!function best_my_matmul(A, x, N) result(b)
!    real, intent(in) :: A(N, N), x(N)
!    integer, intent(in) :: N
!    real :: b(N)
!    
!    integer :: i, j
!
!    b = 0 
!    do j=1, N
!        do i=1, N
!            b(i) = b(i) + A(i,j) * x(j)
!        end do 
!    end do 
!end function




subroutine block_matmul_test(my_matmul)
procedure (interface_matmul_explicit), optional :: my_matmul

    real, allocatable :: A(:, :), x(:), b(:), A_i(:, :), b_i(:)
    integer :: t0, t1, rate
    integer (kind=8) :: N_op, TIMES, N, M, k, i, Q
    real :: R_time(200), theoretical_time

    N_op = 2 * 25e9
    
    !Warmup section
    allocate( A(1000, 1000), x(1000), b(1000) )
    call Initialization(1000, 1000, A, x)
    do k = 0, 20000
        b = matmul(A, x)
    end do
    write(*,*) "b_1 (warmup) = ", b(1)
    write(*,*) "b_N (warmup) = ", b(1000)
    write(*,*)
    deallocate(A, x, b)
    
    N = 5000; M = N
    Q = 125
    allocate(A(N,M), x(M), b(N), A_i(N/Q, N), b_i(N/Q))
    TIMES = N_op / ( 2 * N * M )
    
    write(*,*) "_________________________________________________________________"
    write(*,*) " Ax = b matmul with N = ", N 
    write(*,*) " N_operations = ", N_op
    write(*,*) " TIMES = ", TIMES + 1
    call Initialization(N, M, A, x)

    call system_clock(count_rate=rate)
    call system_clock(t0)
    
    do i = 0, Q-1
        A_i = A(i*N/Q+1:(i+1)*N/Q, :)
        do k = 0, TIMES
            b_i = matmul(A_i, x)
        end do
        b(i*N/Q+1:(i+1)*N/Q) = b_i
    end do
    
    call system_clock(t1)
    
    call print_CPU_time(single_precision = .true., start_time = t0, N_operations = N_op, N_threads = 1)
    write(*,*) "b_1 = ", b(1)
    write(*,*) "b_N = ", b(N)
    deallocate( A, x, b, A_i, b_i )
    theoretical_time = 4 * N_op / ( 2.1e9 * 16 * 1)
    
    R_time = (t1 - t0) / real(rate) / theoretical_time

end subroutine



subroutine explicit_shape_matmul_single_core_static_memory(my_matmul) 
procedure (interface_matmul_explicit), optional :: my_matmul

    integer :: k, t0
    integer :: TIMES = 25000
    integer (kind=8) N_op
     
    N_op = 2 * N * M * real(TIMES)
    
    write(*,*) "_______________________________________"
    write(*,*) " Ax = b matmul with N = ", N 
    write(*,*) " TIMES = ", TIMES
    call Initialization(N, M, A, x)  
   
    call system_clock(t0)
    do k=0, TIMES
        if (present(my_matmul)) then
            b = my_matmul(N, A, x) 
        else
            b = matmul(A, x)
        end if
    end do
    call print_CPU_time(single_precision = .true., start_time = t0, N_operations = N_op, N_threads = 1) 
   
end subroutine


subroutine explicit_shape_single_core_allocation_different_dimension(my_matmul) 
procedure (interface_matmul_explicit), optional :: my_matmul 

    real, allocatable :: A(:, :), x(:), b(:)
    integer :: t0 
    integer (kind=8) :: N_op, TIMES, N, M, k  

    N_op = 2 * 25e9  
    
    !do N = 1000, 5000, 1000
    
    N = 5000
        M = N 
        allocate( A(N,M), x(M), b(N) )
        
        TIMES = N_op / ( 2 * N * M ) 
        
        write(*,*) "________________________________________________________________"
        write(*,*) " Ax = b matmul with N = ", N 
        write(*,*) " N_operations = ", N_op
        call Initialization(N, M, A, x) 
    
        call system_clock(t0)
        do k=0, TIMES  
            if (present(my_matmul)) then 
               b = my_matmul(N, A, x) 
            else 
               b = matmul(A, x)
            end if 
        end do 
        call print_CPU_time(single_precision = .true., start_time = t0, N_operations = N_op, N_threads = 1)
        deallocate( A, x, b )
        
    !end do
        
end subroutine

subroutine explicit_shape_different_dimension_plots(my_matmul)
procedure (interface_matmul_explicit), optional :: my_matmul

    real, allocatable :: A(:, :), x(:), b(:)
    integer :: t0, t1, rate
    integer (kind=8) :: N_op, TIMES, N, M, k, i
    real :: R_time(50), theoretical_time

    N_op = 2 * 25e9  
    
    i = 0.
    do N = 10, 10, 200
    
    !N=5000
        M = N
        i = i + 1.
        allocate( A(N,M), x(M), b(N) )
        
        TIMES = N_op / ( 2 * N * M ) 
        
        write(*,*) "________________________________________________________________"
        write(*,*) " Ax = b matmul with N = ", N 
        write(*,*) " N_operations = ", N_op
        call Initialization(N, M, A, x) 
    
        call system_clock(count_rate=rate)
        call system_clock(t0)
        do k=0, TIMES
            if (present(my_matmul)) then 
                b = my_matmul(N, A, x) 
            else 
                b = matmul(A, x)
            end if 
        end do
        call system_clock(t1)
        
        call print_CPU_time(single_precision = .true., start_time = t0, N_operations = N_op, N_threads = 1)
        deallocate( A, x, b )
        theoretical_time = 4 * N_op / ( 4e9 * 16 * 1)
        
        R_time(i) = (t1 - t0) / real(rate) / theoretical_time
        
    end do
        
    write(*,*)
    write(*,*) "[", R_time, "]"

end subroutine


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



end module








!
!subroutine matmul_single_core_allocation_different_dimension
!        
! real, allocatable :: A(:, :), x(:), b(:)
! integer :: t0 
! integer (kind=8) :: N_op
! integer :: TIMES, N, M, k  
!  
!    
!    N_op = 2 * 25e9  
!    
!    do N = 1000, 5000, 1000
!        M = N 
!        allocate( A(N,M), x(M), b(N) ) 
!   
!        TIMES = N_op / ( 2 * N * M ) 
!        
!    
!        write(*,*) "________________________________________________________________"
!        write(*,*) " Ax = b matmul with N = ", N 
!        write(*,*) " N_operations = ", N_op
!        call Initialization(N, M, A, x) 
!    
!        call system_clock(t0)
!        do k=0, TIMES  
!            b = matmul(A, x) 
!        end do 
!        call print_CPU_time(single_precision = .true., start_time = t0, N_operations = N_op, N_threads = 1)
!        deallocate( A, x, b ) 
!   end do 
!        
!   
!  
!        
!end subroutine
!



!
!
!subroutine test_my_matmul_single_core_static_memory 
!        
! 
! integer :: k, t0
!  
!    integer :: TIMES = 25000
!    integer (kind=8) N_op
!     
!    N_op = 2 * N * M * real(TIMES)
!    
!    write(*,*) "_______________________________________"
!    write(*,*) " Ax = b matmul with N = ", N 
!    write(*,*) " TIMES = ", TIMES
!    call Initialization(N, M, A, x)  
!   
!    call system_clock(t0)
!    do k=0, TIMES  
!            b = my_matmul(A, x) 
!    end do 
!    call print_CPU_time(single_precision = .true., start_time = t0, N_operations = N_op, N_threads = 1) 
!
!  
!        
!end subroutine




!
!subroutine test_my_matmul_single_core_allocation_different_dimension
!        
! real, allocatable :: A(:, :), x(:), b(:)
! integer :: t0 
! integer (kind=8) :: N_op, TIMES, N, M, k  
!  
!    
!    N_op = 2 * 25e9  
!    
!    do N = 1000, 5000, 1000
!        M = N 
!        allocate( A(N,M), x(M), b(N) ) 
!   
!        TIMES = N_op / ( 2 * N * M ) 
!        
!    
!        write(*,*) "________________________________________________________________"
!        write(*,*) " Ax = b matmul with N = ", N 
!        write(*,*) " N_operations = ", N_op
!        call Initialization(N, M, A, x) 
!    
!        call system_clock(t0)
!        do k=0, TIMES  
!            b = my_matmul(A, x) 
!        end do 
!        call print_CPU_time(single_precision = .true., start_time = t0, N_operations = N_op, N_threads = 1)
!        deallocate( A, x, b ) 
!   end do 
!        
!   
!  
!        
!end subroutine






!
!
!
!subroutine test_matmul_single_core( N_operations ) 
! integer(kind=8), intent(out) :: N_operations 
!        
! !real, allocatable :: A(:, :), x(:), b(:)
! integer :: k
! integer :: it1
!  
!    integer :: TIMES = 25000
!   ! allocate( A(N,N), x(N), b(N) ) 
!   
!    N_operations = 2 * N * M * real(TIMES)
!    
!    
!    call Initialization(N, M, A, x) 
! 
!   
!    call system_clock(it1)
!    do k=0, TIMES  
!            b = matmul(A, x) 
!           !write(*,*) k
!    end do 
!    call final_cpu_time1("matmul", it1, b) 
!   
!        
!    call system_clock(it1)
!    do k=0, TIMES  
!        b = my_matmul(A, x) 
!    !   write(*,*) k
!    end do 
!   call final_cpu_time1("my_matmul", it1, b) 
!    
!   write(*,*) " Press enter" 
!   read(*,*)
!
!  
!        
!end subroutine


    
    
    
    
!subroutine test_matmul_single_core( N_operations ) 
! integer(kind=8), intent(out) :: N_operations 
!        
! !real, allocatable :: A(:, :), x(:), b(:)
! integer :: k
! integer :: it1
!  
!    integer :: TIMES = 25000
!   ! allocate( A(N,N), x(N), b(N) ) 
!   
!    N_operations = 2 * N * M * real(TIMES)
!    
!    
!    call Initialization(N, M, A, x) 
! 
!   
!    call system_clock(it1)
!    do k=0, TIMES  
!            b = matmul(A, x) 
!           !write(*,*) k
!    end do 
!    call final_cpu_time1("matmul", it1, b) 
!   
!        
!    call system_clock(it1)
!    do k=0, TIMES  
!        b = my_matmul(A, x) 
!    !   write(*,*) k
!    end do 
!   call final_cpu_time1("my_matmul", it1, b) 
!    
!   write(*,*) " Press enter" 
!   read(*,*)
!
!  
!        
!end subroutine




!
!subroutine  final_cpu_time1(test, it1, b)
!       character(len=*), intent(in) :: test 
!       integer, intent(in) :: it1
!       real, intent(in) :: b(:) 
!  
!    integer :: it2, rate  
!  
!   
!    call system_clock(count_rate=rate)
!    call system_clock(it2)
!    
!    write(*,'(A18, A)') "----- Testing... ", test 
!
!    write(*,*) " CPU_time = ", real(it2-it1)/rate 
!    write(*,*) "b_1 = ", b(1) 
!    write(*,*) "b_M = ", b( size(b) ) 
!    write(*,*) 
!    
!end subroutine     




  ! 
  !call test_matmul_single_core(Nop)  
  !
  !
  !call CPU_specifications_and_compiler(N_operations = Nop, single_precission =.true., parallel = .false.) 
  !
  !stop  
  
!   Ms = 2.**34 
  
!   call test_vectorized_and_parallelized_series
 !  call CPU_specifications_and_compiler(N_threads = 1, N_operations = 3*Ms, single_precission =.false., parallel = .false. ) 
   
!   stop
 !  call sum_series_with_different_threads(Ms) 
  ! call CPU_specifications_and_compiler(N_threads = N_operations = Nop, single_precission =.false., parallel = .true. ) 

 
  
   !write(*,*) "Press enter"
   !read(*,*) 