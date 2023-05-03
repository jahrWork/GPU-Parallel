module functions 
      implicit none 
      
      
contains 
    
real function add_fortran(a, b) 
  real, intent(in) :: a, b 
  
  
   add_fortran =  a + b 

end function 
 
end module 