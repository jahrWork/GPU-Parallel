using LinearAlgebra
using CUDA
using BenchmarkTools
using Plots  # Para graficar

# Función para calcular GFLOPS
function gflops_mul(A, B, C, time_secs)
    n = size(A, 1)
    return (2 * n^3) / (time_secs * 1e9)  # GFLOPS
end

# Benchmark para BLAS (CPU)
function benchmark_blas(N_range)
    println("BLAS (CPU) Benchmark:")
    gflops_results = []
    for N in N_range
        A = rand(N, N)
        B = rand(N, N)
        C = zeros(N, N)
        
        # Warm up
        C .= A * B

        # Time the matrix multiplication
        @btime mul!($C, $A, $B)

        # Measure the performance
        time = @belapsed mul!($C, $A, $B)
        gflops = gflops_mul(A, B, C, time)
        push!(gflops_results, gflops)
        println("Tamaño: $N x $N, GFLOPS: $gflops")
    end
    return gflops_results
end

# Benchmark para CUDA (GPU)
function benchmark_cuda(N_range)
    println("CUDA (GPU) Benchmark:")
    gflops_results = []
    for N in N_range
        A_gpu = CUDA.rand(Float32, N, N)
        B_gpu = CUDA.rand(Float32, N, N)
        C_gpu = CUDA.zeros(Float32, N, N)  # Inicializar C_gpu antes de los bloques de tiempo

        # Warm up
        C_gpu .= A_gpu * B_gpu

        # Time the matrix multiplication
        @btime CUDA.@sync $C_gpu .= $A_gpu * $B_gpu

        # Medir el rendimiento
        time = @belapsed CUDA.@sync $C_gpu .= $A_gpu * $B_gpu
        gflops = gflops_mul(A_gpu, B_gpu, C_gpu, time)
        push!(gflops_results, gflops)
        println("Tamaño: $N x $N, GFLOPS: $gflops")
    end
    return gflops_results
end

# Rango de tamaños de matrices
N_range = 200:200:5000

# Ejecutar benchmarks
blas_gflops = benchmark_blas(N_range)
cuda_gflops = benchmark_cuda(N_range)

# Mostrar resultados
println("\nResultados de GFLOPS para BLAS: ", blas_gflops)
println("Resultados de GFLOPS para CUDA: ", cuda_gflops)

# Graficar resultados
plot(N_range, blas_gflops, label="BLAS (CPU)", marker=:circle, linewidth=2)
plot!(N_range, cuda_gflops, label="CUDA (GPU)", marker=:diamond, linewidth=2)
xlabel!("Tamaño de la Matriz (N)")
ylabel!("GFLOPS")
title!("Comparación de GFLOPS: BLAS vs CUDA (Multiplicación Matriz-Matriz)")
