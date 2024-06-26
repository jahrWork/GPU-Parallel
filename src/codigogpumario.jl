using CUDA
using CUDA.CUBLAS
using Plots
import BenchmarkTools: @belapsed



const THREADS_PER_BLOCK = (16, 16)

# Initialize matrices on GPU
function matrix_initialization_GPU(N::Int)
    A = CUDA.rand(Float32, N, N)
    B = CUDA.rand(Float32, N, N)
    return A, B
end

# Manual GPU matrix multiplication kernel
function matrix_multiplication_GPU(A, B)
    C = CUDA.zeros(eltype(A), size(A, 1), size(B, 2))
    threads_per_block = (16, 16)
    blocks_per_grid = (ceil(Int, size(A, 1) / threads_per_block[1]), ceil(Int, size(B, 2) / threads_per_block[2]))

    @cuda blocks=blocks_per_grid threads=threads_per_block matrix_mul_kernel(A, B, C)
    return C
end

# CUDA kernel for matrix multiplication
function matrix_mul_kernel(A, B, C)
    tx = threadIdx().x
    ty = threadIdx().y
    bx = blockIdx().x
    by = blockIdx().y
    bw = blockDim().x
    bh = blockDim().y

    x = (bx - 1) * bw + tx
    y = (by - 1) * bh + ty

    if x <= size(A, 1) && y <= size(B, 2)
        sum = 0f0
        for k in 1:size(A, 2)
            sum += A[x, k] * B[k, y]
        end
        C[x, y] = sum
    end
    return
end

# cuBLAS GPU matrix multiplication
function matrix_multiplication_cublas(A, B)
    C = CUDA.zeros(eltype(A), size(A, 1), size(B, 2))
    alpha = 1f0
    beta = 0f0
    CUBLAS.gemm!('N', 'N', alpha, A, B, beta, C)
    return C
end

# Benchmark both methods
function benchmark_multiplication(N_range)
    times_manual = Float64[]
    times_cublas = Float64[]
    for N in N_range
        A, B = matrix_initialization_GPU(N)
        time_manual = @belapsed matrix_multiplication_GPU($A, $B)
        time_cublas = @belapsed matrix_multiplication_cublas($A, $B)
        push!(times_manual, time_manual)
        push!(times_cublas, time_cublas)
        println("N = $N: Manual = $(time_manual), cuBLAS = $(time_cublas)")
    end
    return times_manual, times_cublas
end

# Main script to run benchmark and plot results
function main()
    N = 100:100:2500
    times_manual, times_cublas = benchmark_multiplication(N)
    if length(times_manual) > 0 && length(times_cublas) > 0
        GFLOPS_manual = 1 ./ (times_manual * 1e-9)
        GFLOPS_cublas = 1 ./ (times_cublas * 1e-9)

        plot(N, GFLOPS_manual, label="Manual GFLOPS", title="GFLOPS Comparison", ylims=(0, 200), xlabel="Matrix Size N", ylabel="GFLOPS")
        plot!(N, GFLOPS_cublas, label="cuBLAS GFLOPS")
    end
end

main()