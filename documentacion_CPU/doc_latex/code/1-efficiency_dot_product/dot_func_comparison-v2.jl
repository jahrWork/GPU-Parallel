import Pkg
Pkg.activate(".")
Pkg.add( "PGFPlotsX" )
using CPUTime
using Plots
using LinearAlgebra, MKL
using PGFPlotsX

#Function to initialize random matrices
function matrix_initialization(N)

    A = rand(Float32, N, N )
    B = rand(Float32, N, N )
    return A, B 

end 

# Function to multiply matrices using the built-in Julia method
function matrix_multiplication(A,B)

   return A * B  
  
end

# Function to multiply matrices using a custom method (manual loop)
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

# Function to time matrix multiplication and calculate performance
function time_matrix_multilication(N, N_cores, matmul)

  Time = zeros( length(N) )

  for (i,n) in enumerate(N)
 
   A,B = matrix_initialization(n)

   t1= time_ns()

   matmul(A,B)
   
   t2 = time_ns()
   Time[i] = (t2-t1)/(2*n^3)
  
   #println("N=", n, " Time per operation =", Time[i] , " nsec")
  end 
  
  return Time

end 

# settings "julia.NumThreads": "auto"
N_threads = Threads.nthreads()
N_cores =  div(N_threads, 2)
println("Threads =", N_threads ) 
println("Cores =", N_cores )

# Precompilation: Run matrix multiplication once to warm up
time_matrix_multilication(2000, N_cores, matrix_multiplication)

# Set range for matrix dimensions
N = 0:100:2500

# Set number of threads for BLAS operations (used by matrix multiplication)
BLAS.set_num_threads(2*N_cores)
println(" threads = ", BLAS.get_num_threads(), " N_cores =", N_cores )

# Time the built-in matrix multiplication and custom multiplication
Time = time_matrix_multilication(N, N_cores, matrix_multiplication)
Time2 = time_matrix_multilication(N, N_cores, my_matrix_multiplication)

# Calculate GFLOPS (floating point operations per second) for each method
GFLOPS = 1 ./ Time
GFLOPS2 = 1 ./ Time2

# Data for plotting
x = N
y1 = GFLOPS
y2 = GFLOPS2

# Create the plot using PGFPlotsX
plot = @pgf Axis(
    {
        xlabel="Matrix dimension [N]",
        ylabel="GFLOPS",
        title="Comparison of different dot functions",
        #legend="north east"
    },
    Plot({no_marks, "blue"}, Table(x, y1)),
    Plot({no_marks, "red"}, Table(x, y2)),
)

PGFPlotsX.save("/Users/juanromanbermejo/Desktop/documentacion_CPU/doc_latex/code/1-efficiency_dot_product/grafico_dot_func_comparison.tex", plot, include_preamble=false)
