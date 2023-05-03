!  Fortran.f90 
!
!  FUNCTIONS:
!  Fortran - Entry point of console application.
!

!****************************************************************************
!
!  PROGRAM: Fortran
!
!  PURPOSE:  Entry point for the console application.
!
!****************************************************************************

    
    
    program main
    
        
    use functions 
    implicit none
 
    interface 
      function  add_cpp(a, b) bind(c)
          use, intrinsic :: iso_c_binding 
          real (c_float) :: a, b 
          real (c_float) :: add_cpp
      end function 
    end interface
    
    write( *, *) 'Fortran sum =', add_fortran(3.4, 4.3)
    write( *, *) 'C++ sum =', add_cpp(3.4, 4.3) 

    end program 

