nimport Pkg
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

function matrixvector_initialization(N)


  A = rand(Float32, N, N )
  B = rand(Float32, N, 1 )

  return A, B 

end 


function matrix_multiplication(A,B)

  return  A * B  

end



function time_matrix_vector_multilication(N, N_cores, matinit, matmul)

  Time2 = zeros( length(N) )
  #Theoretical_time = 4e9/(4e9 * 512/32 * 4 * N_cores)
  Theoretical_time = 1e9/(4e9 * 512/32 *  N_cores) # jahr

  for (i,n) in enumerate(N)  # variables inside loop have local scope 
 
   A,Av = matinit(n)
   #m = length(B)

   

  #  dt = 1e9 * matmul(A,B)
   
  #  CUDA.unsafe_free!(A)
  #  CUDA.unsafe_free!(B)

   t1 = time_ns()
   for _ in 1:n
   matmul(A,Av)
   end
   t2 = time_ns()
   dt = t2-t1
   
   Time2[i] = dt/(2*n^3)
   

   println("N=", n, " Time per operation =", Time2[i] , " nsec")
   println("N=", n, " Theoretical time per operation =", Theoretical_time, " nsec")
    
  end 
  
  return Time2

end 

function time_matrix_multilication(N, N_cores, matinit, matmul)

    Time = zeros( length(N) )
    #Theoretical_time = 4e9/(4e9 * 512/32 * 4 * N_cores)
    Theoretical_time = 1e9/(4e9 * 512/32 *  N_cores) # jahr
  
    for (i,n) in enumerate(N)  # variables inside loop have local scope 
   
     A,B = matinit(n)
     #m = length(B)
  
     
  
    #  dt = 1e9 * matmul(A,B)
     
    #  CUDA.unsafe_free!(A)
    #  CUDA.unsafe_free!(B)
  
     t1 = time_ns()
     matmul(A,B)
     t2 = time_ns()
     dt = t2-t1
     
     Time[i] = dt/(2*n^3)
     
  
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

  combined_plot = plot(title="GFLOPS different N_cores", xlabel="N", ylabel="GFLOPS", ylims=(0, 350), minorgrid=true)
  
  for N_threads in N_threads_range
      N_threads = N_threads 
      N_cores = N_threads 
      println("Threads =", N_threads ) 
      println("Cores =", N_cores ) 
      
      N = Vector(10:100:5000)
      BLAS.set_num_threads(N_cores) 
      println(" threads = ", BLAS.get_num_threads(), " N_cores =", N_cores )
      Time, Theoretical_time = time_matrix_multilication(N, N_cores, matrix_initialization, matrix_multiplication)
      Time2 = time_matrix_vector_multilication(N, N_cores, matrixvector_initialization, matrix_multiplication)
      GFLOPS = 1 ./ Time
      GFLOPS2 = 1 ./ Time2
      GFLOPS_max = 1 / Theoretical_time

      plot!(combined_plot, N, GFLOPS, label="Threads = $N_threads, AxB", legend=:topleft)
      plot!(combined_plot, N, GFLOPS2, label="Threads = $N_threads AxAv, N times", legend=:topleft)
      plot!(combined_plot, N, GFLOPS_max * ones(length(N)), label=false, linecolor=:black)
  end
  display(combined_plot)
end

plot_combined(4)








