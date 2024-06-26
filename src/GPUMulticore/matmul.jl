import Pkg
Pkg.activate(".")  # environment in this folder
Pkg.add( "Plots" ) 
Pkg.add( "CPUTime" ) 
#Pkg.add( "BLIS" )
Pkg.add( "MKL" )
#Pkg.precompile()


using CPUTime
using Plots


#using LinearAlgebra, BLIS
using LinearAlgebra, MKL


function matrix_initialization(N)

  
    A = rand(Float32, N, N )
    B = rand(Float32, N, N )

    return A, B 

end 


function matrix_multiplication(A,B)

     return A * B  
  
end

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




function time_matrix_multilication(N, N_cores, matmul)

  Time = zeros( length(N) )
  #Theoretical_time = 4e9/(4e9 * 512/32 * 4 * N_cores)
  Theoretical_time = 1e9/(4e9 * 512/32  * N_cores) # jahr

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

# settings "julia.NumThreads": "auto"
N_threads = 6
#N_cores =  div(N_threads, 2)
N_cores = N_threads
println("Threads =", N_threads ) 
println("Cores =", N_cores ) 

time_matrix_multilication(2000, N_cores, matrix_multiplication) # precompilation
time_matrix_multilication(2000, N_cores, my_efficient_matrix_multiplication) 
time_matrix_multilication(2000, N_cores, my_efficient_matrix_multiplication2) 
time_matrix_multilication(2000, N_cores, my_matrix_multiplication) 



N = Vector([10:10:2500; 2500:100:10000])
BLAS.set_num_threads(2*N_cores)
println(" threads = ", BLAS.get_num_threads(), " N_cores =", N_cores )
Time, Theoretical_time = time_matrix_multilication(N, N_cores, matrix_multiplication)
GFLOPS = 1 ./ Time
GFLOPS_max = 1 / Theoretical_time


display( plot(N, Time, ylims=(1e-4, 1e-2), yaxis=:log, minorgrid=true  ) )
display( plot!(N, Theoretical_time*ones( length(N) ), yaxis=:log, minorgrid=true ) )

xlabel!("N")
display( plot(N, GFLOPS, ylims=(0, 1000), title="GFLOPS",  minorgrid=true  ) )
display( plot!(N, GFLOPS_max *ones( length(N) ), minorgrid=true ) )


