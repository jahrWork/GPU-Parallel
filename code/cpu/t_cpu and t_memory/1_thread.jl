using CPUTime, LinearAlgebra, MKL, BenchmarkTools

N = 50
Nt = 1000
A = rand(Float32, N, N)
B = rand(Float32, N, N)
C = zeros(Float32, size(A, 1), size(B, 2))

t_CPU = ( 2 * N^3 / (432/6 *10^9) ) * 10^6
t_memory = ( 3 * N^2 / (3200 * 10^6) ) * 10^6

println("=====================================")
println(" ")
println("MxM Benchmark 1 thread")
println("t_CPU: ", t_CPU, " μs")
println("t_memory: ", t_memory, " μs")
println(" ")

# Set BLAS to single-thread mode
BLAS.set_num_threads(1)

function matrix_mult(A, B)
    for i in 1:Nt
        C = A * B
    end
end


# Benchmark the matrix multiplication
dt1 = @belapsed C = A * B
dt2 = ( @belapsed matrix_mult(A, B) ) / Nt

# println("t1: ", t1)
# println("t2: ", t2)
# println("dt: ", dt)
println("Time 1: ", dt1 * 10^6, " μs - GFLOPS: ", 2 * N^3 / dt1 / 10^9)
println("Time 2: ", dt2 * 10^6, " μs - GFLOPS: ", 2 * N^3 / dt2 / 10^9)



# # Obtain the median allocations and memory usage
# median_allocations = median(benchmark_result).allocs
# median_memory = round(median(benchmark_result).memory / 1024, digits=2)
    
# # Print allocations and memory metrics
# println("Allocations: ", median_allocations)
# println("Memory allocated: ", median_memory, " KB")