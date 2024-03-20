import Pkg
Pkg.add( "Plots" ) 
Pkg.add( "CPUTime" ) 
#Pkg.add( "BLIS" )
Pkg.add( "MKL" )
#Pkg.precompile()

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

function matrix_multiplication_row_by_column(A,B)

  (N, M) = size(A)
  (M, L) = size(B) 

  C = zeros(Float32,  (N, L) )

  
  for i in 1:N
    for j in 1:L 
      for k in 1:M
        C[i,j] = C[i,j] + A[i,k]*B[k,j]
      end  
    end
  end

  return C 

end

function matrix_multiplication_column_by_row(A,B)

  (N, M) = size(A)
  (M, L) = size(B) 

  C = zeros(Float32,  (N, L) )

  for j in 1:L 
    for i in 1:N
      for k in 1:M
        C[i,j] = C[i,j] + A[i,k]*B[k,j]
      end  
    end
  end

  return C 

end



function time_for_different_N(N, N_cores, matmul)

  Time = zeros( length(N) )
  Theoretical_time = 4e9/(4e9 * 512/32 * 4 * N_cores)

  for (i,n) in enumerate(N)  # variables inside loop have local scope 
 
   A,B = matrix_initialization(n)

   t1= time_ns()

   matmul(A,B)
   
   t2 = time_ns()
   Time[i] = (t2-t1)/(2*n^3)
   

   println("N=", n, " Time per operation =", Time[i] , " nsec")
   println("N=", n, " Theoretical time per operation =", Theoretical_time, " nsec")
    
  end 
  
  return Time, Theoretical_time

end 


N_cores = 1 
time_for_different_N(2000, N_cores, matrix_multiplication) # precompilation
time_for_different_N(2000, N_cores, matrix_multiplication_row_by_column) 
time_for_different_N(2000, N_cores, matrix_multiplication_row_by_column) 






#using LinearAlgebra, BLIS
using LinearAlgebra, MKL

N = Vector([10:10:2500; 2500:100:6000])
N_cores = 4 
BLAS.set_num_threads(2*N_cores)
println(" threads = ", BLAS.get_num_threads() )
Time, Theoretical_time = time_for_different_N(N, N_cores, matrix_multiplication)
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

