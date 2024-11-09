using CPUTime, LinearAlgebra, MKL, BenchmarkTools

N = 50
Nt = 1000
A = rand(Float32, N, N)
B = rand(Float32, N, 1)
C = zeros(Float32, size(A, 1), size(B, 2))

t_CPU = ( 2 * N^2 / (432/6 * 10^9) ) * 10^6
t_memory = ( ( N^2 + 2N) / (3200 * 10^6) ) * 10^6

println("=====================================")
println(" ")
println("MxV")
println("t_CPU: ", t_CPU, " μs")
println("t_memory: ", t_memory, " μs")
println(" ")

# Set BLAS to single-thread mode
BLAS.set_num_threads(1)

# warm up
C = A * B

function matrix_vector_mult(A, B)
    for i in 1:Nt
        C = A * B
    end
end


# Benchmark the matrix multiplication
dt1 = @belapsed C = A * B
dt2 = ( @belapsed matrix_vector_mult(A, B) ) / Nt

# println("t1: ", t1)
# println("t2: ", t2)
# println("dt: ", dt)
println("Time 1: ", dt1 * 10^6, " μs")
println("Time 2: ", dt2 * 10^6, " μs")



# # Obtain the median allocations and memory usage
# median_allocations = median(benchmark_result).allocs
# median_memory = round(median(benchmark_result).memory / 1024, digits=2)
    
# # Print allocations and memory metrics
# println("Allocations: ", median_allocations)
# println("Memory allocated: ", median_memory, " KB")

# Deberes:
# 1. @belapsed en MKL_levels.jl
# 2. Partir los Nt en los diferentes threads con CpuId
# 3. Uniformizar los plots de CPU y GPU 
# 4. Ejemplo práctico (ecuación de ondas 2D) Matrix-Vector vs Matrix-Matrix (CPU y GPU)