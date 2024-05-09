import Pkg
Pkg.activate(".")  # environment in this folder
Pkg.add( "Plots" ) 
Pkg.add( "CPUTime" ) 
#Pkg.add( "BLIS" )
Pkg.add( "MKL" )
Pkg.add("CUDA")

#Pkg.precompile()


using CPUTime
using Plots


#using LinearAlgebra, BLIS
#using LinearAlgebra, MKL
using LinearAlgebra
using CUDA





function my_matrix_multiplication(A,B)

  (N, M) = size(A)
  (M, L) = size(B) 

  C = zeros(Float32,  (N, L) )

  
  for i in 1:N, j in 1:L
      for k in 1:M
        C[i,j] = C[i,j] + A[i,k]*B[k,j]
      end  
  end

  return C 

end

function my_efficient_matrix_multiplication(A,B)

  (N, M) = size(A)
  (M, L) = size(B) 
  BT = transpose(B) 

  C = zeros(Float32,  (N, L) )

  for k in 1:M
    for j in 1:L, i in 1:N
      
        C[i,j] = C[i,j] + A[i,k]*BT[j,k]

    end  
  end

  return C 

end

function my_efficient_matrix_multiplication2(A,B)

  (N, M) = size(A)
  (M, L) = size(B) 
  BT = transpose(B) 

  C = zeros(Float32,  (N, L) )
  

  Threads.@threads for k in 1:M
    for j in 1:L, i in 1:N
      
        C[i,j] = C[i,j] + A[i,k]*BT[j,k]
        
    end  
  end

  return C 

end














function matrix_initialization_GPU(N)

  # N = 10000

  A = rand(Float32, N, N)
  B = rand(Float32, N, N)

  d_A = CUDA.CuArray(A)
  d_B = CUDA.CuArray(B)
  
  return d_A, d_B 
 
end 

function matrix_initialization(N)


  A = rand(Float32, N, N )
  B = rand(Float32, N, N )

  return A, B 

end 


function matrix_multiplication2(A,B)

   return @elapsed A * B  

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




function plot_results(GFLOPS, GFLOPS_max, title, ymax)

  # display( plot(N, Time, ylims=(1e-4, 1e-2), yaxis=:log, minorgrid=true  ) )
  # display( plot!(N, Theoretical_time*ones( length(N) ), yaxis=:log, minorgrid=true ) )
 
   xlabel!("N")
   display( plot(N, GFLOPS, ylims=(0, ymax), title= title,  minorgrid=true  ) )
   display( plot!(N, GFLOPS_max *ones( length(N) ), minorgrid=true ) )
 
 end 


 function plot_combined(N_threads_range)

  combined_plot = plot(title="GFLOPS different N_cores", xlabel="N", ylabel="GFLOPS", ylims=(0, 2000), minorgrid=true)
  
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

#plot_combined(1:4)






# # settings "julia.NumThreads": "auto"
# N_threads = Threads.nthreads()
# N_cores =  div(N_threads, 2)

# println("Threads =", N_threads ) 
# println("Cores =", N_cores ) 

# # time_matrix_multilication(2000, N_cores, matrix_initilization, matrix_multiplication) # precompilation
# # time_matrix_multilication(2000, N_cores, matrix_initilization, my_efficient_matrix_multiplication) 
# # time_matrix_multilication(2000, N_cores, matrix_initilization, my_efficient_matrix_multiplication2) 
# # time_matrix_multilication(2000, N_cores, matrix_initilization, my_matrix_multiplication) 


#  N_cores = 4 
#  N = Vector([10:10:2500; 2500:100:5000])
#  BLAS.set_num_threads(N_cores)
#  println(" threads = ", BLAS.get_num_threads(), " N_cores =", N_cores )
#  Time, Theoretical_time = time_matrix_multilication(N, N_cores, matrix_initialization, matrix_multiplication)
#  GFLOPS_CPU = 1 ./ Time
#  GFLOPS_max = 1 / Theoretical_time

# plot_results(GFLOPS_CPU, GFLOPS_max, "GFLOPS CPU", 2000)



# N = Vector([10:10:2500; 2500:1000:5000])
# BLAS.set_num_threads(N_cores)
# println(" threads = ", BLAS.get_num_threads(), " N_cores =", N_cores )
# Time, Theoretical_time = time_matrix_multilication(N, N_cores, matrix_initialization_Av, matrix_multiplication)

# GFLOPS_CPU = 1 ./ Time
# GFLOPS_max = 1 / Theoretical_time

# plot_results(GFLOPS_CPU, GFLOPS_max, "GFLOPS Av CPU", 2000)


# N = Vector(10:10:2500)
# N_cores = 1
# BLAS.set_num_threads(N_cores)
# println(" threads = ", BLAS.get_num_threads(), " N_cores =", N_cores )
# Time, Theoretical_time = time_matrix_multilication(N, N_cores, matrix_initialization, matrix_multiplication)
# GFLOPS_CPU = 1 ./ Time
# GFLOPS_max = 1 / Theoretical_time

# plot_results(GFLOPS_CPU, GFLOPS_max, "GFLOPS CPU", 200)




N_cores = 8

N = Vector(10:1000:10000)
Time, Theoretical_time = time_matrix_multilication(N, N_cores, matrix_initialization_Av, matrix_multiplication)
GFLOPS_CPU = 1 ./ Time
GFLOPS_max = 1 / Theoretical_time

plot_results(GFLOPS_CPU, GFLOPS_max, "GFLOPS CPU", 1000)




# N_cores = 1

# N = Vector(10:10:10000)
# Time, Theoretical_time = time_matrix_multilication(N, N_cores, matrix_initialization_GPU, matrix_multiplication_GPU)
# GFLOPS_GPU = 1 ./ Time
# GFLOPS_max = 1 / Theoretical_time

# plot_results(GFLOPS_GPU, GFLOPS_max, "GFLOPS GPU", 40000)







