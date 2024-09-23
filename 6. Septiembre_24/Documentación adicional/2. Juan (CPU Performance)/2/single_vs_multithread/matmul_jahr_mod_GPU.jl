import Pkg
Pkg.activate(".")  # environment in this folder
Pkg.add( "Plots" ) 
Pkg.add( "MKL" )
Pkg.add("CUDA")

using Plots
using LinearAlgebra
using CUDA

########################################################
############## INICIALIZACIÓN DE MATRICES ##############
########################################################

function matrix_initialization_GPU(N)
  # N = 10000

  A = rand(Float32, N, N)
  B = rand(Float32, N, N)

  d_A = CUDA.CuArray(A)
  d_B = CUDA.CuArray(B)
  
  return d_A, d_B 
 
end 

############## CÁLCULO DE MATMUL ##############
function matrix_multiplication(A,B)

  return  A * B  

end

function matrix_multiplication_GPU(A,B)

  
  return CUDA.@elapsed A * B  
  
  end

function time_matrix_multilication(N, N_cores, matinit, matmul)

  Time = zeros( length(N) )
  #Theoretical_time = 4e9/(4e9 * 512/32 * 4 * N_cores)
  Theoretical_time = 1e9/(4e9 * 512/32 *  N_cores) # jahr

  for (i,n) in enumerate(N)  # variables inside loop have local scope 
 
   A,B = matinit(n)
   m = length(B)

   

  #  dt = 1e9 * matmul(A,B)
   
  #  CUDA.unsafe_free!(A)
  #  CUDA.unsafe_free!(B)

   t1 = time_ns()
   matmul(A,B)
   t2 = time_ns()
   dt = t2-t1
   
   Time[i] = dt/(2*n*m)
   

   println("N=", n, " Time per operation =", Time[i] , " nsec")
   println("N=", n, " Theoretical time per operation =", Theoretical_time, " nsec")
    
  end 
  
  return Time, Theoretical_time

end 

function plot_results(GFLOPS, GFLOPS_max, title, ymax)

  # display( plot(N, Time, ylims=(1e-4, 1e-2), yaxis=:log, minorgrid=true  ) )
  # display( plot!(N, Theoretical_time*ones( length(N) ), yaxis=:log, minorgrid=true ) )
 
   xlabel!("N")
   display( plot(N, GFLOPS, ylims=(0, ymax), title= title,  minorgrid=true  ) )
   display( plot!(N, GFLOPS_max *ones( length(N) ), minorgrid=true ) )
 
 end 


 N_cores = 1
 N = Vector(10:10:10000)
 Time, Theoretical_time = time_matrix_multilication(N, N_cores, matrix_initialization_GPU, matrix_multiplication_GPU)
 GFLOPS_GPU = 1 ./ Time
 GFLOPS_max = 1 / Theoretical_time

 max_GFLOPS_GPU = maximum(GFLOPS_GPU)
 label = @sprintf("Max GFLOPS: %.2f", max_GFLOPS_GPU)

 plot_results(GFLOPS_GPU, GFLOPS_max, "GFLOPS GPU", 40000, label)






