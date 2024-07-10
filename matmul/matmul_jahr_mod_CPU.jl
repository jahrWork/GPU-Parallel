import Pkg
Pkg.activate(".")  # environment in this folder
#Pkg.add( "Plots" ) 
#Pkg.add( "CPUTime" ) 
#Pkg.add( "BLIS" )
Pkg.add( "MKL" )
#Pkg.add("CUDA")

#Pkg.precompile()


using CPUTime
using Plots


#using LinearAlgebra, BLIS
using LinearAlgebra, MKL
#using LinearAlgebra
#using CUDA

function matrix_initialization(N)


  A = rand(Float32, N, N )
  B = rand(Float32, N, N )

  return A, B 

end 


function matrix_multiplication2(A,B)

   return @elapsed A * B  

end

function matrix_add2(A,B)

  return @elapsed A + B  

end

function vector_multiplication2(A,B)

  return @elapsed dot(A, B)  

end


function matrix_multiplication(A,B)

  return  A * B  

end

        function matrix_initialization_Av(N)


        A = rand(Float32, N, N )
        v = rand(Float32, N, N )

        return A, v 

        end 

        function matrix_multiplication_Av(A, v)

        
        Time = zeros( Float32, length(v)  )

        for i in 1:length(v)
            
            b[i] = dot( A[i, :], v )

        end 

        return b 

        end 


#======================================#
#======================================#

#====== MATRIX x MATRIX ======#

N = 20000

A, B = matrix_initialization(N)

time_AxB = matrix_multiplication2(A,B)
GFLOPS_AxB = 1e-9 / (time_AxB/(2*N^3))


#====== MATRIX x VECTOR ======#

C = rand(Float32, N, 1 )
D = rand(Float32, 1, N )

Av = rand(Float32, N )
Bv = rand(Float32, N )

time_AxC = matrix_multiplication2(A,C)
GFLOPS_AxC = 1e-9 / (time_AxC/(2*N^2))

time_AxD = matrix_multiplication2(D,A)
GFLOPS_AxD = 1e-9 / (time_AxD/(2*N^2))

time_AxAv = matrix_multiplication2(A,Av)
GFLOPS_AxAv = 1e-9 / (time_AxAv/(2*N^2))


#===== VECTOR x VECTOR =====#

time_AvxBv = vector_multiplication2(Av,Bv)
GFLOPS_AvxBv = 1e-9 / (time_AvxBv/(2*N))

time_DxC = vector_multiplication2(D,C)
GFLOPS_DxC = 1e-9 / (time_DxC/(2*N))


#===== MATRIX + MATRIX =====#

time_A_add_B = matrix_add2(A,B)
GFLOPS_A_add_B = 1e-9 / (time_A_add_B/(N^2))


#======================================#
#======================================#
#======================================#


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


 function plot_combined(N_threads_range)

  combined_plot = plot(title="GFLOPS different N_cores", xlabel="N", ylabel="GFLOPS", ylims=(0, 300), minorgrid=true)
  
  for N_threads in N_threads_range
      N_threads = N_threads 
      N_cores = N_threads 
      println("Threads =", N_threads ) 
      println("Cores =", N_cores ) 
      
      N = Vector([10:10:2500; 2500:100:5000])
      BLAS.set_num_threads(N_cores) 
      println(" threads = ", BLAS.get_num_threads(), " N_cores =", N_cores )
      Time, Theoretical_time = time_matrix_multilication(N, N_cores, matrix_initialization, matrix_multiplication)
      GFLOPS = 1 ./ Time
      GFLOPS_max = 1 / Theoretical_time

      plot!(combined_plot, N, GFLOPS, label="Threads = $N_threads", legend=:topleft)
      plot!(combined_plot, N, GFLOPS_max * ones(length(N)), label=false, linecolor=:black)
  end
  display(combined_plot)
end

plot_combined(1:4)








