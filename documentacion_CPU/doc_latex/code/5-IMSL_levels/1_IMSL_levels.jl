import Pkg
Pkg.activate(".")
Pkg.add( "PGFPlotsX" ) 
using CPUTime
using Plots
using LinearAlgebra, MKL
using PGFPlotsX

# Function to initialize random matrices of size N x N
function matrix_initialization(N)


  A = rand(Float32, N, N )
  B = rand(Float32, N, N )

  return A, B 

end 

# Function to initialize a random matrix and a vector (N x 1)
function matrix_vector_initialization(N)


    A = rand(Float32, N, N )
    B = rand(Float32, N, 1 )
  
    return A, B 
  
end 

# Function for vector multiplication (dot product)
function vector_multiplication(A,B)

    return dot(A, B)  
  
end

# Function for matrix multiplication
function matrix_multiplication(A,B)

    return  A * B  

end

# Function to time matrix multiplication operations
function time_matrix_multilication(N, N_cores, matinit, matmul)

    Time = zeros( length(N) )
    Theoretical_time = 1e9/(4e9 * 512/32 *  N_cores)
  
    for (i,n) in enumerate(N)  
   
     A,B = matinit(n)
  
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
  
# Function to time matrix-vector multiplication operations
function time_matrix_vector_multilication(N, N_cores, matinit, matmul)

    Time2 = zeros( length(N) )
  
    for (i,n) in enumerate(N) 
   
     A,B = matinit(n)
    
     t1 = time_ns()
     matmul(A,B)
     t2 = time_ns()
     dt = t2-t1
     
     Time2[i] = dt/(2*n^2)
  
     println("N=", n, " Time per operation =", Time2[i] , " nsec")
     println("N=", n, " Theoretical time per operation =", Theoretical_time, " nsec")
      
    end 
    
    return Time2
  
end 

# Number of cores
N_cores = 4

# Range of matrix dimensions to test
N = Vector([10:25:2500; 2500:100:5000])

# Set the number of BLAS threads based on the number of cores
BLAS.set_num_threads(2*N_cores) 
println(" threads = ", BLAS.get_num_threads(), " N_cores =", N_cores )

# Time the matrix multiplication and matrix-vector multiplication operations
Time, Theoretical_time = time_matrix_multilication(N, N_cores, matrix_initialization, matrix_multiplication)
Time2 = time_matrix_vector_multilication(N, N_cores, matrix_vector_initialization, matrix_multiplication)

# Calculate GFLOPS (floating-point operations per second)
GFLOPS = 1 ./ Time
GFLOPS2 = 1 ./ Time2
GFLOPS_max = 1 / Theoretical_time

# Data for plotting
x = N
y1 = GFLOPS
y2 = GFLOPS2


plot = @pgf Axis(
    {
        xlabel="Matrix dimension",
        ylabel="FLOPS [GFLOPS]",
        title="[M]x[M] vs [M]x[v]",
        #legend="north east"
    },
    Plot({no_marks, "blue"}, Table(x, y1)),
    Plot({no_marks, "red"}, Table(x, y2)),
)

PGFPlotsX.save("/Users/juanromanbermejo/Desktop/documentacion_CPU/doc_latex/code/5-IMSL_levels/1_IMSL_levels.tex", plot, include_preamble=false)





