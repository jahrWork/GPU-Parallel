using Plots
using LinearAlgebra
using CUDA

########################################################
############## INITIALIZATION OF MATRICES ##############
########################################################

function matrix_initialization_GPU(N)
  # N = 10000

  A = rand(Float32, N, N)  # Initialize matrix A with random values
  B = rand(Float32, N, N)  # Initialize matrix B with random values

  d_A = CUDA.CuArray(A)  # Transfer matrix A to GPU
  d_B = CUDA.CuArray(B)  # Transfer matrix B to GPU
  
  return d_A, d_B 
end 

############## MATRIX MULTIPLICATION ##############
function matrix_multiplication(A, B)
  return A * B  # Perform matrix multiplication on CPU
end

function matrix_multiplication_GPU(A, B)
  return CUDA.@elapsed A * B  # Measure time taken for matrix multiplication on GPU
end

function time_matrix_multiplication(N, N_cores, matinit, matmul)
  Time = zeros(length(N))  # Array to store computation times
  Theoretical_time = 1e9 / (4e9 * 512/32 * N_cores)  # Theoretical time per operation

  for (i, n) in enumerate(N)  # Loop through each matrix size
    A, B = matinit(n)  # Initialize matrices
    m = length(B)

    t1 = time_ns()  # Start time measurement
    matmul(A, B)  # Perform matrix multiplication
    t2 = time_ns()  # End time measurement
    
    dt = t2 - t1
    Time[i] = dt / (2 * n * m)  # Calculate time per operation

    println("N=", n, " Time per operation =", Time[i], " nsec")
    println("N=", n, " Theoretical time per operation =", Theoretical_time, " nsec")
  end 
  
  return Time, Theoretical_time
end 

function plot_results(GFLOPS, GFLOPS_max, title, ymax)
  xlabel!("N")  # Label x-axis
  display(plot(N, GFLOPS, ylims=(0, ymax), title=title, minorgrid=true))  # Plot GFLOPS
  display(plot!(N, GFLOPS_max * ones(length(N)), minorgrid=true))  # Plot theoretical GFLOPS
end 

N_cores = 1
N = Vector(10:10:10000)  # Define matrix sizes
Time, Theoretical_time = time_matrix_multiplication(N, N_cores, matrix_initialization_GPU, matrix_multiplication_GPU)
GFLOPS_GPU = 1 ./ Time  # Compute GFLOPS
GFLOPS_max = 1 / Theoretical_time  # Compute maximum theoretical GFLOPS

plot_results(GFLOPS_GPU, GFLOPS_max, "GFLOPS GPU", 6000)