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
    threads_per_block = (16, 16)  # Define the number of threads per block
    blocks_per_grid = (ceil(Int, size(C, 1) / threads_per_block[1]), 
                       ceil(Int, size(C, 2) / threads_per_block[2]))  # Calculate blocks per grid
    CUDA.@cuda threads=threads_per_block blocks=blocks_per_grid mul_kernel(C, A, B, size(C, 1), size(C, 2), size(B, 2))
    return C
end

function mul_kernel(C, A, B, M, N, P)
    i, j = threadIdx().x + (blockIdx().x - 1) * blockDim().x, threadIdx().y + (blockIdx().y - 1) * blockDim().y
    if i <= M && j <= P
        acc = zero(eltype(C))
        for k = 1:N
            acc += A[i, k] * B[k, j]
        end
        C[i, j] = acc
    end
    return
end

function time_matrix_multiplication_GPU(N_range)
times = Float64[]
theoretical_times = Float64[]

for N in N_range
    A, B = matrix_initialization_GPU(N)
    ops_per_iteration = 2 * N^2
    dt = @belapsed matrix_multiplication_GPU($A, $B)  # Pass A and B as arguments to @belapsed macro  
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

N = 10:500:2500
CUDA.allowscalar(false)  

# GPU times and GFLOPS
times_GPU, theoretical_times_GPU = time_matrix_multiplication_GPU(N)
GFLOPS_GPU = 1 ./ times_GPU
GFLOPS_max_GPU = 1 / theoretical_time()  

# Graphics
plot_results(N, GFLOPS_GPU, GFLOPS_max_GPU, "GFLOPS GPU", 1000)