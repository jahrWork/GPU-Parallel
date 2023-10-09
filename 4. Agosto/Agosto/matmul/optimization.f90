 module optimization 
    
   use CPU_specifications  
   implicit none 
   integer, parameter :: N = 1000, M= 1000
   real, save :: A(N, M), x(M), b(N)
   private 
   public :: test_optimization
   
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
    
    interface 
     function interface_C_AXB(N, A, B) result(C) 
      integer, intent(in) :: N
      real, intent(in) :: A(N,N), B(N,N) 
      real :: C(N,N)
     end function
   end interface 
    
   
    contains   
    
subroutine test_optimization

   integer :: option 
   
   do
     write(*,*) " Select an option " 
     write(*,*) " 0. Exit / Quit "
     write(*,*) " 1. Loop / Array Ordering " 
     write(*,*) " 2. Indirect Addressing " 
     write(*,*) " 3. Repeated Array Accesses " 
     write(*,*) " 4. Array Layout" 
     write(*,*) " 5. Array Allocation " 
     write(*,*) " 6. Array Slicing "
     write(*,*) " 7. Array Temporaries "
     write(*,*) " 8. Register Spilling "
     write(*,*) " 9. Floating Point Issues "
     write(*,*) " 10. NPROMA Slicing "
     
     read(*,*) option   
     
     select case(option)
        
     case(0) 
          exit
          
     case(1)
         call loop_array_ordering
         
     case(2)
         call indirect_addressing
         
     case(3)
         call repeated_array_accesses
         
    case(4)
         call array_layout
         
     case(5)
         call array_allocation
         
     case(6)
         call array_slicing
         
     case(7)
         call array_temporaries
         
     case(8)
         call register_spilling
         
     case(9)
         call floating_point_issues
         
     case(10) 
         call nproma_slicing
         
     case default
         write(*,*) "Invalid option, please try again."
         
     end select      
     
   end do

end subroutine 
 




subroutine loop_array_ordering
    implicit none
    integer(kind=8), parameter :: nx=100, ny=100, nz=100 
    real, dimension(nx, ny, nz) :: a, b, c
    integer(kind=8) :: i, j, k, repetition
    integer(kind=8), parameter :: nreps = 200000
    integer :: t0, t1
    integer(kind=8) :: N_op
    
    write(*,*) " "
    write(*,*) "You selected loop_array_ordering"
    write(*,*) " "
    
    ! Initialize arrays with random numbers
    call random_number(a)
    call random_number(b)
    call random_number(c)
    
    N_op = nx * ny * nz * nreps

    ! Bad time (k-j-i)
    call system_clock(t0)
    do repetition = 1, nreps
        do k = 1, nz
            do j = 1, ny
                do i = 1, nx
                    a(k, j, i) = b(k, j, i) + c(k, j, i)
                end do
            end do
        end do
    end do

    call print_CPU_time(single_precision = .true., start_time=t0, N_operations = N_op, N_threads = 1)
    
    
    ! Reset array to initial state
    call random_number(a)
    call random_number(b)
    call random_number(c)
    
    ! Good time (i-j-k)
    call system_clock(t0)
    do repetition = 1, nreps
        do i = 1, nx
            do j = 1, ny
                do k = 1, nz
                    a(i, j, k) = b(i, j, k) + c(i, j, k)
                end do
            end do
        end do
    end do
    
   call print_CPU_time(single_precision = .true., start_time=t0, N_operations = N_op, N_threads = 1)
      
   
end subroutine






subroutine indirect_addressing

    implicit none
    integer(kind=8), parameter :: nelems=100, num_neighbors=100, nlevels=100
    real, dimension(nelems, num_neighbors, nlevels) :: elem_val, neighbor_val
    integer(kind=8):: ie, i, k, repetition
    integer(kind=8), parameter :: nreps = 200000
    integer :: t0, t1
    integer(kind=8) :: N_op
    
    N_op = nelems * num_neighbors * nlevels * nreps
    
    write(*,*) " "
    write(*,*) "You selected indirect_addressing"
    write(*,*) " "

    ! Initialization of arrays for testing purposes
    call random_number(elem_val)
    call random_number(neighbor_val)

    ! Bad Example
    call system_clock(t0)
    do repetition = 1, nreps
        do ie = 1, nelems
            do i = 1, num_neighbors
                elem_val(:, ie, i) = elem_val(:, ie, i) + neighbor_val(:, ie, i)
            end do
        end do
    end do
    
    write(*,*) "Bad ordering time:"
    call print_CPU_time(single_precision=.true., start_time=t0, N_operations=N_op, N_threads=1)

    ! Reset array to initial state
    call random_number(elem_val)
    call random_number(neighbor_val)

    ! Good Example
    call system_clock(t0)
    do repetition = 1, nreps
        do ie = 1, nelems
            do i = 1, num_neighbors
                do k = 1, nlevels
                    elem_val(k, ie, i) = elem_val(k, ie, i) + neighbor_val(k, ie, i)
                end do
            end do
        end do
    end do
    
    write(*,*) "Good ordering time:"
    call print_CPU_time(single_precision=.true., start_time=t0, N_operations=N_op, N_threads=1)

end subroutine 






subroutine repeated_array_accesses
    implicit none
    integer(kind=8), parameter :: np = 100
    real, dimension(np, np) :: matrix, dat1, dat2, matrix_i, dat1_i, dat2_i
    integer(kind=8) :: i, j, l, t0, repetition
    real :: tmp
    integer(kind=8), parameter :: nreps = 1000
    integer(kind=8) :: N_op
    
    N_op = 2 * np * np * np * nreps

    ! Initialization of arrays for testing purposes
    call random_number(matrix)
    call random_number(dat1)
    
    matrix_i = matrix
    dat1_i = dat1

    
        
        ! Bad Example
        write(*,*) "Bad"
        call system_clock(t0)

        do repetition = 1, nreps
            do j = 1 , np
                do l = 1 , np
                dat2(l,j) = 0
                    do i = 1 , np
                        dat2(l,j) = dat2(l,j) + matrix(i,l)*dat1(i,j)
                    end do
                end do
            end do
        end do
    
        call print_CPU_time(single_precision = .true., start_time = t0, N_operations = N_op, N_threads = 1) 
        

        ! Good Example
        dat2_i = 0
        write(*,*) "Good"
        call system_clock(t0)
        do repetition = 1, nreps
            do j = 1 , np
                do l = 1 , np
                tmp = 0
                    do i = 1 , np
                        tmp = tmp + matrix_i(i,l)*dat1_i(i,j)
                    end do
                dat2_i(l,j) = tmp
                end do
            end do
        end do
        
        call print_CPU_time(single_precision = .true., start_time = t0, N_operations = N_op, N_threads = 1) 
    
end subroutine




subroutine array_layout
    implicit none
    integer(kind=8), parameter :: np=10000
    real, dimension(n) :: array, dat1, dat2, tmp, array_i, dat1_i, dat2_i, tmp_i
    integer(kind=8) :: i, repetition, t0, t1, b, c, N_op
    integer(kind=8), parameter :: nreps = 1000000
     

    N_op = 2 * np * np * np * nreps
    
    
     call random_number(array)
     call random_number(dat1)
     call random_number(dat2)
     b = 1
     c = 1
     
     array_i = array
     dat1_i = dat1
     dat2_i = dat2
     
     ! Bad example with loop-carried dependence
     
    write(*,*) "Bad array_layout"
    
    call system_clock(t0)
    
    do repetition = 1, nreps
        !array(1) = dat1(1)**b
            do i = 2, n
             array(i) = array(i-1) + dat1(i)**b
             dat2(i) = array(i)**c
            end do
    end do

    call print_CPU_time(single_precision = .true., start_time = t0, N_operations = N_op, N_threads = 1) 

    !do i = 1, n
    !    write(*,*) 'dat2(i) bad: ', dat2(i)
    !end do


    ! Good example without loop-carried dependence
    
    write(*,*) "Good array_layout"
    
    call system_clock(t0)
    
    do repetition = 1, nreps
        !tmp_i(1) = dat1_i(1)**b
        do i = 2, n
            tmp_i(i) = dat1_i(i)**b
        end do
        !array_i(1) = tmp_i(1)
        do i = 2, n
            array_i(i) = array_i(i-1) + tmp_i(i)
        end do
        do i = 2, n
            dat2_i(i) = array_i(i)**c
        end do
    end do
    
    call print_CPU_time(single_precision = .true., start_time = t0, N_operations = N_op, N_threads = 1) 
    
    read(*,*)
    
    do i = 2, n
        write(*,*) 'dat2(i) bad: ', dat2(i)
    end do
    
    do i = 2, n
         write(*,*) 'dat2(i) good: ', dat2_i(i)
    end do


    !! Bad example with loop-carried dependence
    !call system_clock(t0)
    !do repetition = 1, nreps
    !    do i = 1, n
    !        array(i) = array(i-1) + dat1(i)**b
    !        dat2(i) = array(i)**c
    !    end do
    !end do
    !
    !call print_CPU_time(single_precision = .true., start_time = t0, N_operations = N_op, N_threads = 1) 
    !!call system_clock(t1)
    !!bad_time = t1 - t0
    !
    !do i = 1, n
    !    write(*,*) 'dat2(i) bad: ', dat2(i)
    !end do
    !
    !
    !! Good example without loop-carried dependence
    !call system_clock(t0)
    !do repetition = 1, nreps
    !    do i = 1, n
    !        tmp_i(i) = dat1_i(i)**b
    !    end do
    !    do i = 1, n
    !        array_i(i) = array_i(i-1) + tmp_i(i)
    !    end do
    !    do i = 1, n
    !        dat2_i(i) = array_i(i)**c
    !    end do
    !end do
    !
    !call print_CPU_time(single_precision = .true., start_time = t0, N_operations = N_op, N_threads = 1) 
    !!call system_clock(t1)
    !!good_time = t1 - t0
    !
    !do i = 1, n
    !    write(*,*) 'dat2(i) good: ', dat2_i(i)
    !end do
    
    !write(*,*) 'Bad time: ', bad_time
    !write(*,*) 'Good time: ', good_time

end subroutine



subroutine array_allocation

    ! Información sobre el heap
    write(*,*) '-----------------------------------------------'
    write(*,*) 'Heap:'
    write(*,*) 'La memoria del heap está disponible para todo el programa y se asigna dinámicamente.'
    write(*,*) 'Puedes decidir durante la ejecución cuánta memoria quieres asignar.'
    write(*,*) 'Sin embargo, asignar y desasignar memoria del heap es un proceso relativamente costoso en términos de tiempo de ejecución.'
    write(*,*) 'En Fortran, los arrays globales y los arrays asignados con el comando "allocate()" se almacenan en el heap.'
    write(*,*) '-----------------------------------------------'
    
    ! Información sobre el stack
    write(*,*) 'Stack:'
    write(*,*) 'La memoria del stack es más limitada, pero la asignación y desasignación de memoria es muy rápida.'
    write(*,*) 'Los arrays locales declarados en una subrutina se almacenan en el stack.'
    write(*,*) '-----------------------------------------------'

    ! Información sobre la interacción entre arrays y threads
    write(*,*) 'Threads:'
    write(*,*) 'Cada thread tiene su propio stack, pero todos los threads comparten el heap.'
    write(*,*) 'Si un array en el stack se crea dentro de una región paralela, cada thread tendrá su propia copia.'
    write(*,*) 'Todos los threads pueden acceder a los arrays en el heap, por lo que si varios threads necesitan modificar un array en el heap, será necesario algún tipo de sincronización.'
    write(*,*) '-----------------------------------------------'

    ! Recomendaciones
    write(*,*) 'Recomendaciones:'
    write(*,*) '1. Asignar arrays dinámicos sólo a un alto nivel e infrecuentemente.'
    write(*,*) '2. Usar arrays en el stack para todos los temporales de array de subrutina.'
    write(*,*) '3. Entender el tamaño de estos arrays para tener una buena estimación del stacksize necesario.'
    write(*,*) '-----------------------------------------------'
    
end subroutine


subroutine array_slicing
    implicit none
    integer(kind=8), parameter:: n=1000
    real, dimension(n,n) :: a, b, c, d, f, g
    integer(kind=8):: i, ii
     integer :: t0, t1
    integer(kind=8), parameter :: nreps = 100000
    real :: start, finish, elapsed
    integer (kind=8) :: N_op, repetition
    
    N_op = 3*n*n*nreps

    ! Initialize arrays with random numbers
    call random_number(a)
    call random_number(b)
    call random_number(c)
    call random_number(d)
    call random_number(f)

    ! Example with array slicing (not optimal)
    write(*,*) "Execution time with array slicing (not optimal)"
    call system_clock(t0)
    do repetition = 1, nreps
        do i = 1, n
            b(:,i) = b(:,i) * c(:,i)
            a(:,i) = a(:,i) + b(:,i)
            a(:,i) = a(:,i) + d(:,i)
        end do
    end do
    call print_CPU_time(single_precision = .true., start_time = t0, N_operations = N_op, N_threads = 1)
    
    ! Reset arrays to initial state
    call random_number(a)
    call random_number(b)
    call random_number(c)
    call random_number(d)
    call random_number(f)

    ! Example without array slicing (optimal)
    write(*,*) "Execution time without array slicing (optimal)"
    call system_clock(t0)
    do repetition = 1, nreps
        do i = 1, n
            do ii = 1, n
                b(ii,i) = b(ii,i) * c(ii,i)
                a(ii,i) = a(ii,i) + b(ii,i)
                a(ii,i) = a(ii,i) + d(ii,i)
            end do
        end do
    end do
    call print_CPU_time(single_precision = .true., start_time = t0, N_operations = N_op, N_threads = 1)
    
end subroutine



subroutine array_temporaries

end subroutine


subroutine register_spilling

end subroutine


subroutine floating_point_issues

end subroutine


subroutine nproma_slicing
    implicit none
    integer :: n, m, k
    real, dimension(:,:,:), allocatable :: a, b, c, d, z
    real :: startTime, endTime, timeArraySyntax, timeNproma
    integer :: j, jm, jk

    ! Asignamos valores a n, m y k
    n = 600
    m = 600
    k = 600

    ! Asignamos los arrays
    allocate(a(n,m,k), b(n,m,k), c(n,m,k), d(n,m,k), z(n,m,k))

    ! Inicialización de los arrays 
    a = 1.0
    b = 2.0
    c = 3.0
    d = 4.0
    z = 5.0

    ! Comenzamos a medir el tiempo usando la sintaxis de array
    call cpu_time(startTime)
    do j = 1, n
        a(j,:,:) = a(j,:,:)*z(j,:,:)
        b(j,:,:) = a(j,:,:)**2.
        c(j,:,:) = c(j,:,:)+b(j,:,:)
        d(j,:,:) = d(j,:,:)/3.+c(j,:,:)
        z(j,:,:) = z(j,:,:)*d(j,:,:)
    enddo
    call cpu_time(endTime)

    timeArraySyntax = endTime - startTime

    print*, 'Array Syntax Result: '
    write(*,*) 'a(1,1,1)=', a(1,1,1)
    write(*,*) 'b(2,2,2)=', b(2,2,2)
    write(*,*) 'c(3,3,3)=', c(3,3,3)
    write(*,*) 'd(4,4,4)=', d(4,4,4)
    write(*,*) 'z(5,5,5)=', z(5,5,5)

    ! Reiniciamos los arrays
    a = 1.0
    b = 2.0
    c = 3.0
    d = 4.0
    z = 5.0

    ! Comenzamos a medir el tiempo usando nproma slicing
    call cpu_time(startTime)
    do jk = 1, k
        do jm = 1, m
            do j = 1, n
                a(j,jm,jk) = a(j,jm,jk)*z(j,jm,jk)
                b(j,jm,jk) = a(j,jm,jk)**2.
                c(j,jm,jk) = c(j,jm,jk)+b(j,jm,jk)
                d(j,jm,jk) = d(j,jm,jk)/3.+c(j,jm,jk)
                z(j,jm,jk) = z(j,jm,jk)*d(j,jm,jk)
            enddo
        enddo
    enddo
    call cpu_time(endTime)

    timeNproma = endTime - startTime

    print*, 'Nproma Slicing Result: '
    write(*,*) 'a(1,1,1)=', a(1,1,1)
    write(*,*) 'b(2,2,2)=', b(2,2,2)
    write(*,*) 'c(3,3,3)=', c(3,3,3)
    write(*,*) 'd(4,4,4)=', d(4,4,4)
    write(*,*) 'z(5,5,5)=', z(5,5,5)

    print*, 'Tiempo con la sintaxis de array: ', timeArraySyntax
    print*, 'Tiempo con nproma slicing: ', timeNproma

    ! No olvides desasignar los arrays al final
    deallocate(a, b, c, d, z)

end subroutine nproma_slicing







    end module
    
