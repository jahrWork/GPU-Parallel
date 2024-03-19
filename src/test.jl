import Pkg
Pkg.add( "Plots" ) 
Pkg.add( "CPUTime" ) 
Pkg.add( "BLIS" )
Pkg.add( "MKL" )
Pkg.precompile()

using CPUTime
using Plots

function matrix_initialization(N)

  
    A = rand(Float32, N, N )
    B = rand(Float32, N, N )

    return A, B 

end 


function matrix_multiplication(A,B)

     return A * B  
  
end


function time_for_different_N(N, N_cores)

  Time = zeros( length(N) )
  Theoretical_time = 4e9/(4e9 * 512/32 * 4 * N_cores)

  for (i,n) in enumerate(N)  # variables inside loop are local scope 
 
   A,B = matrix_initialization(n)

   t1= time_ns()

   matrix_multiplication(A,B)
   
   t2 = time_ns()
   Time[i] = (t2-t1)/(2*n^3)
   

   println("N=", n, " Time per operation =", Time[i] , " nsec")
   println("N=", n, " Theoretical time per operation =", Theoretical_time, " nsec")
    
  end 
  
  return Time, Theoretical_time

end 



time_for_different_N(200) # precompilation


N = Vector([10:10:2000; 2000:100:6000])



#using LinearAlgebra, BLIS
using LinearAlgebra, MKL

N_cores = 4 
BLAS.set_num_threads(2*N_cores)
println(" threads = ", BLAS.get_num_threads() )
Time, Theoretical_time = time_for_different_N(N, N_cores)
display( plot(N, Time, ylims=(0., 0.05) ) )
display( plot!(N, Theoretical_time*ones( length(N) ) ) )



# BLAS.set_num_threads(Sys.CPU_THREADS)
# println(" threads = ", BLAS.get_num_threads() )
# Time = time_for_different_N(N)
# plot(N, Time)    


# begin 
# BLAS.set_num_threads(4)
# println(" threads = ", BLAS.get_num_threads() )
# Time, Theoretical_time = time_for_different_N(N)
# plot(N, Time, ylims=(0., 0.05) )
# plot!(N, Theoretical_time*ones( length(N) ) ) 
# end


# ENV["JULIA_NUM_THREADS"] = "4"

