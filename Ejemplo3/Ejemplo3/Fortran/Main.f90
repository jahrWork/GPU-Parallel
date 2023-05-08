program main

    use, intrinsic :: iso_c_binding
    use functions
  
    implicit none

    interface
        subroutine print_C( string ) bind( C, name = "print_C" )
          !use iso_c_binding, only : C_CHAR    !Also valid, priority to user-provided module of the same name, but doesn't exist for now
          use, intrinsic :: iso_c_binding, only : C_CHAR
          character ( kind = C_CHAR ) :: string( * )
        end subroutine 
        
        function add_cpp(a, b) bind(c, name = "add_cpp")
          use, intrinsic :: iso_c_binding, only : C_FLOAT
          real (c_float), value :: a, b 
          real (c_float) :: add_cpp
        end function 
        
        function add_cpp2(a, b) bind(c, name = "add_cpp2")
          use, intrinsic :: iso_c_binding, only : C_INT
          real (c_int), value :: a, b 
          real (c_int) :: add_cpp2
        end function 
        
        function mul_cpp3(a, b) bind(c, name = "mul_cpp3")
          use, intrinsic :: iso_c_binding, only : C_FLOAT
          real (c_float) :: a(2,2), b(2,1) 
          real (c_float) :: mul_cpp3(2,1)
        end function 
    end interface

    
    integer :: r, t
    real :: b(1:4) 
    
    
    b = (/ 1, 2, 3, 4 /)
    
    r = 3
    t = 4
    
    
    
    write(*,*) 'Fortran sum =', add_fortran(3.4, 4.3)
    write(*,*) 'C++ sum =', add_cpp(3.4, 4.3)
    write(*,*) 'C++ sum2 =', add_cpp2(3, 4)
    write(*,*) 'C++ mul =', mul_cpp3( reshape( b, (/ 2, 2 /), order = (/ 2, 1 /) ), (/ 2, 2 /))
    call print_C( C_CHAR_"Hello World!" // C_NULL_CHAR )

 

end