using Plots
using LinearAlgebra
using CUDA

########################################################
####################### GPU INFO #######################
########################################################

function get_architecture_name(capability)
  if capability.major == 9
      return "Ada Lovelace"  # RTX 40 series
  elseif capability.major == 8
      return "Ampere"  # RTX 30 series
  elseif capability.major == 7
      if capability.minor == 5
          return "Turing"  # RTX 20 series
      else
          return "Volta"  # Tesla V100
      end
  elseif capability.major == 6
      return "Pascal"  # GTX 10 series
  elseif capability.major == 5
      return "Maxwell"  # GTX 900 series
  elseif capability.major == 3
      return "Kepler"  # GTX 600/700 series
  else
      return "Unknown Architecture"
  end
end

function cuda_cores_per_sm(capability)
  if capability.major == 9  # Ada Lovelace architecture (e.g., RTX 40 series)
      return 128
  elseif capability.major == 8  # Ampere architecture (e.g., RTX 30 series)
      return 128
  elseif capability.major == 7  # Turing/Volta architecture
      return 64
  elseif capability.major == 6  # Pascal architecture
      return 64
  else
      return 32  # Default to older architectures
  end
end

function gpu_info()
  device = CUDA.device()
  capability = CUDA.capability(device)
  sm_count = CUDA.attribute(device, CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
  cores_per_sm = cuda_cores_per_sm(capability)  # Get cores per SM based on compute capability
  total_cuda_cores = sm_count * cores_per_sm

  println(" ")
  println("---------------------------------")
  println(" ")
  println("Device Information ")
  println(" ")
  println("GPU Name: ", CUDA.name(device))
  println("GPU Compute Capability: ", capability.major, ".", capability.minor, " (", get_architecture_name(capability), ")")
  println(" ")
  println("GPU Memory: ", CUDA.totalmem(device) / 1e9, " GB (base-10, where 1 GB = 1,000,000,000 bytes)")
  println("GPU Memory: ", CUDA.totalmem(device) / 2^30, " GiB (base-2, where 1 GiB = 1,073,741,824 bytes)")
  println(" ")
  println("GPU Streaming Multiprocessor (SM) Count: ", sm_count)
  println("CUDA Cores per SM: ", cores_per_sm)
  println("Total CUDA Cores: ", total_cuda_cores)
  println(" ")
  println("GPU Clock Rate: ", CUDA.attribute(device, CUDA.DEVICE_ATTRIBUTE_CLOCK_RATE) / 1e6, " GHz")
  println(" ")
  println("---------------------------------")
  println(" ")
end


########################################################
############## INITIALIZATION OF MATRICES ##############
########################################################

function matrix_initialization_GPU(N)
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

# Function to plot the results using Plots.jl
function plot_results(N, GFLOPS, GFLOPS_max, title, ymax)
  plot(N, GFLOPS, label="Measured GFLOPS", lw=2, title=title, xlabel="Matrix dimension [N]", ylabel="GFLOPS", legend=:topright, ylim=(0, ymax))
  plot!(N, fill(GFLOPS_max, length(N)), label="Theoretical GFLOPS", lw=2, linestyle=:dash, color=:red)
end 

# Display GPU information
gpu_info()

# Test parameters
N_cores = 1
N = Vector(100:100:15000)  # Define matrix sizes
Time, Theoretical_time = time_matrix_multiplication(N, N_cores, matrix_initialization_GPU, matrix_multiplication_GPU)
GFLOPS_GPU = 1 ./ Time  # Compute GFLOPS
GFLOPS_max = 1 / Theoretical_time  # Compute maximum theoretical GFLOPS

# Create the plot with the Plots.jl library
plot_results(N, GFLOPS_GPU, GFLOPS_max, "Matrix Multiplication Performance (GFLOPS)", maximum(GFLOPS_GPU))
