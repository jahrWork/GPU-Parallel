import Pkg
Pkg.activate(".")
Pkg.add(["PGFPlotsX", "CPUTime", "Plots", "LinearAlgebra", "MKL"])
using CPUTime
using Plots
using LinearAlgebra, MKL
using PGFPlotsX
using CpuId

#CPU info
cpuid = cpuinfo()
string_cpuid = string(cpuid)

println("AVX support: ", occursin("256", string_cpuid))
println("AVX-512 support: ", occursin("512 bit", string_cpuid))

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

# Transposing B for efficient memory access
function my_efficient_matrix_multiplication(A, B)
  (N, M) = size(A)
  (M, L) = size(B)
  BT = transpose(B) 
  C = zeros(Float32, (N, L))

  for k in 1:M
      for j in 1:L, i in 1:N
          C[i, j] = C[i, j] + A[i, k] * BT[j, k]
      end
  end

  return C
end

# Function to time matrix multiplication and calculate performance
function time_matrix_multilication(N, N_cores, matmul, AVX_value)

  Theoretical_time = 1e9 /(4.5e9 * AVX_value * 2 * N_cores)

  Time = zeros( length(N) )

  for (i,n) in enumerate(N)
 
   A,B = matrix_initialization(n)

   t1= time_ns()

   matmul(A,B)
   
   t2 = time_ns()
   Time[i] = (t2-t1)/(2*n^3)
  
   #println("N=", n, " Time per operation =", Time[i] , " nsec")
  end 
  
  return Time, Theoretical_time

end 

function get_avx_value(string_cpuid)
  AVX_value = 0

  if occursin("256 bit", string_cpuid)
      AVX_value = 8
  elseif occursin("512 bit", string_cpuid)
      AVX_value = 16
  else
      AVX_value = 0
  end
  
  return AVX_value
end

AVX_value = get_avx_value(string_cpuid)

# settings "julia.NumThreads": "auto"
# en bash: $ JULIA_NUM_THREADS=4 julia
BLAS.set_num_threads(8)
N_threads = BLAS.get_num_threads()
N_cores =  div(N_threads, 2)
println("Threads =", N_threads ) 
println("Cores =", N_cores )

# Precompilation: Run matrix multiplication once to warm up
time_matrix_multilication(2000, N_cores, matrix_multiplication, AVX_value)

# Set range for matrix dimensions
N = 10:100:2500

# Set number of threads for BLAS operations (used by matrix multiplication)
BLAS.set_num_threads(2*N_cores)
println(" threads = ", BLAS.get_num_threads(), " N_cores =", N_cores )

# Time the built-in matrix multiplication and custom multiplication
Time, Theoretical_time = time_matrix_multilication(N, N_cores, matrix_multiplication, AVX_value)
Time2, Theoretical_time2 = time_matrix_multilication(N, N_cores, my_matrix_multiplication, AVX_value)
Time3, Theoretical_time3 = time_matrix_multilication(N, N_cores, my_efficient_matrix_multiplication, AVX_value)

# Calculate GFLOPS (floating point operations per second) for each method
GFLOPS = 1 ./ Time
GFLOPS2 = 1 ./ Time2
GFLOPS3 = 1 ./ Time3
GFLOPS_max = 1 ./ Theoretical_time

println(typeof(Time2))
println(Time2)
println(typeof(Time3))
println(Time3)

# Data for plotting
x = N
y1 = GFLOPS
y2 = GFLOPS2
y3 = GFLOPS3
y4 = GFLOPS_max
y4_vector = fill(y4, length(y1))

# Create the plot using PGFPlotsX
plot_dot = @pgf Axis(
    {
        width = "15cm",  
        height = "10cm",  
        xlabel="Matrix dimension [N]",
        ylabel="GFLOPS",
        title="Comparison of different matrix multiplication functions",
        legend="north east",
        ymax=500,
        #ymode="log",

    },
    Plot({no_marks, "blue"}, Table(x, y1)),
    Plot({no_marks, "orange"}, Table(x, y2)),
    Plot({no_marks, "green"}, Table(x, y3)),
    Plot({no_marks, "red"}, Table(x, y4_vector)),
    LegendEntry("matrix_multiplication"),
    LegendEntry("my_matrix_multiplication"),
    LegendEntry("my_efficient_matrix_multiplication"),
    LegendEntry("Theoretical GFLOPS"),
)

display(plot_dot)

#PGFPlotsX.save("code/dot_func_comparison_dyn.tex", plot_dot, include_preamble=false)
