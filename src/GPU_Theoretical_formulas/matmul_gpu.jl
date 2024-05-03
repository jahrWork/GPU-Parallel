import Pkg
Pkg.activate(".")
Pkg.add(["CUDA", "Plots", "BenchmarkTools"])

using CUDA
using LinearAlgebra
using Plots
using BenchmarkTools

function theoretical_time()
    return 1e9 / (140e6 * 128/32) 
end

function matrix_initialization_GPU(N::Int)
    A = CUDA.rand(Float32, N, N)
    B = CUDA.rand(Float32, N, N)
    return A, B
end

function matrix_multiplication_GPU(A, B)
    C = CUDA.zeros(eltype(A), size(A, 1), size(B, 2))
    threads_per_block = 16
    blocks_per_grid = ceil(Int, size(C, 1) / threads_per_block)
    @cuda blocks=blocks_per_grid threads=threads_per_block mul_kernel(C, A, B, size(C, 1), size(B, 2))
    return C
end

function mul_kernel(C, A, B, M, N)
    i, j = threadIdx().x + blockDim().x * (blockIdx().x - 1), threadIdx().y + blockDim().y * (blockIdx().y - 1)
    stride_x, stride_y = blockDim().x * gridDim().x, blockDim().y * gridDim().y
    acc = zero(eltype(C))
    for k = 1:N
        acc += A[i, k] * B[k, j]
    end
    C[i, j] = acc
    return
end

function time_matrix_multiplication_GPU(N_range)
    times = Float64[]
    theoretical_times = Float64[]
    
    for N in N_range
        A, B = matrix_initialization_GPU(N)
        ops_per_iteration = 2 * N^2
        dt = @belapsed matrix_multiplication_GPU(A, B)  
        time_per_operation = 1e9 * dt / ops_per_iteration  
        
        push!(times, time_per_operation)
        push!(theoretical_times, theoretical_time())
        
        println("N = $N, Time per operation = $time_per_operation nsec")
        println("N = $N, Theoretical time per operation = $(theoretical_time()) nsec")
    end
    
    return times, theoretical_times
end


function plot_results(N, GFLOPS, GFLOPS_max, title, ymax)
    plot(N, GFLOPS, ylims=(0, ymax), title=title, label="Real GFLOPS", minorgrid=true)
    hline!([GFLOPS_max], label="Theoretical GFLOPS", linestyle=:dash, color=:red)
    xlabel!("Matrix Size N")
    ylabel!("GFLOPS")
end

N = 10:10:2500
CUDA.allowscalar(false)  

# GPU times and GFLOPS
times_GPU, theoretical_time_GPU = time_matrix_multiplication_GPU(N)
GFLOPS_GPU = 1 ./ times_GPU
GFLOPS_max_GPU = 1 / theoretical_time()  

# Graphics
plot_results(N, GFLOPS_GPU, GFLOPS_max_GPU, "GFLOPS GPU", 5000)
