using LinearAlgebra, MKL
using Plots
 
N = 10_000
A = rand(Float32, N, N)
B = rand(Float32, N, N)
 
function matrix_multiplication(A, B)
    return A * B
end
 
N_threads = [1, 2, 3, 4, 5, 6]
times = Float64[]
 
matrix_multiplication(A, B)
 
BLAS.set_num_threads(1)
reference_time = @elapsed matrix_multiplication(A, B)
 
# Calculando el speedup para cada número de hilos
speedups = Float64[]
for (i, threads) in enumerate(N_threads)
    BLAS.set_num_threads(threads)
        
    t = @elapsed matrix_multiplication(A, B)
    push!(times, t)
    speedup = reference_time / t
    push!(speedups, speedup)
    println("Threads: $threads, Time: $t, Speedup: $speedup")
end
 
plot(N_threads, speedups, xlabel="Number of Threads (N_threads)", ylabel="Speedup", title="Matrix Multiplication Speedup vs. Number of Threads", legend=false)