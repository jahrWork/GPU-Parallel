!  C_AxB.f90 
!
!  FUNCTIONS:
!  C_AxBFortran - Entry point of console application.
!

!****************************************************************************
!
!  PROGRAM: C_AxB
!
!  PURPOSE:  Entry point for the console application.
!
!****************************************************************************

    program C_AxB
    
    use omp_lib

    implicit none
    
    real, allocatable :: A(:,:), B(:,:), C(:,:)
    real :: power, dimensions(139)
    integer :: j, i, unitNum
    integer :: rate, t0, t1
    character(len=250) :: fileName
    integer (kind=8) :: N_ops, N, TIMES
    real :: execution_time
    
    
    N_ops = 2 * 10000_8**3;
    write(*,*) "CPU TEST RUNNING with N_ops = ", N_ops
    
    ! Crea la carpeta "resultados" (ignorará el error si ya existe)
    call execute_command_line("if not exist Resultados mkdir Resultados")
    
    ! Combina la ruta de la carpeta con la cadena de fecha y hora
    fileName = "resultados/" // "results_octubre_multi" // ".csv"
    ! Abre el archivo CSV para escritura
    open(unit=unitNum, file=fileName, status='unknown')  
    !write(unitNum,*) "N,TIMES,Execution Time (s)"
    
    ! Warm up section
    N = 5000; TIMES = 2
    allocate(A(N, N), B(N, N), C(N, N))
    call random_number(A)
    call random_number(B)
    C = 0
    do i = 1, TIMES
        call sgemm("N", "N", N, N, N, 1e0, A, N, B, N, 0, C, N)
    end do
    deallocate(A, B, C)
    
    
    do N = 50, 2499, 25
        
        TIMES = nint(N_ops / real(2*N**3, 8))
        
        allocate(A(N, N), B(N, N), C(N, N))
        
        call random_number(A)
        call random_number(B)
        
        C = 0
        
        call system_clock(t0)
        call system_clock(count_rate=rate)
        
        !$omp parallel do private(i)
        do i = 1, TIMES
            call sgemm("N", "N", N, N, N, 1e0, A, N, B, N, 0, C, N)
        end do
        !$omp end parallel do
        
        call system_clock(t1)
        execution_time = (t1 - t0) / real(rate)
        write(unitNum,*) N, TIMES, execution_time
        write(*,*) N, TIMES, execution_time
        deallocate(A, B, C)
        
    end do
    
    
    
    ! Shortest loop
    do TIMES = 64, 1, -1
        
        power = 1./3
        N = nint((real(10000_8**3) / TIMES)**power)
        dimensions(TIMES) = N
        
        
        allocate(A(N, N), B(N, N), C(N, N))
        
        call random_number(A)
        call random_number(B)
        
        C = 0
        
        call system_clock(t0)
        call system_clock(count_rate=rate)
        
        !$omp parallel do private(i)
        do i = 1, TIMES
            call sgemm("N", "N", N, N, N, 1e0, A, N, B, N, 0, C, N)
        end do
        !$omp end parallel do
        
        call system_clock(t1)
        execution_time = (t1 - t0) / real(rate)
        write(unitnum,*) n, times, execution_time
        write(*,*) n, times, execution_time
        
        deallocate(A, B, C)
        
    end do
    
    close(unitNum)
    
    read(*,*)
    
    
    
    end program C_AxB
