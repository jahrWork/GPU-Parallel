using CUDA
using BenchmarkTools
import LinearAlgebra.BLAS: vendor

function benchmark_matrix_multiplication(filename, n, num_runs)
    Time_GPU = zeros(num_runs)
    
    open(filename, "a") do file
        for i in 1:num_runs
            A = rand(Float32, n, n)
            B = rand(Float32, n, n)

            d_A = CUDA.CuArray(A)
            d_B = CUDA.CuArray(B)
            
            CUDA.@sync d_A * d_B

            total_time_GPU = @belapsed CUDA.@sync d_A * d_B  evals=1
            
            write(file, "$n, $(2 * n^3), $(vendor()), $total_time_GPU\n")
            
            Time_GPU[i] = total_time_GPU
        end
    end
    
    return Time_GPU
end

filename = "benchmark_results_cuda.csv"
N = 50:50:1450
num_runs = 10

Time_GPU = zeros(length(N), num_runs)

for (i, n) in enumerate(N)
    Time_GPU[i, :] = benchmark_matrix_multiplication(filename, n, num_runs)
end

using Statistics
mean_Time_GPU = mean(Time_GPU, dims=2)

plot(N, mean_Time_GPU,
    xlabel="Matrix Size", ylabel="Mean Execution Time (s)",
    title="Matrix Multiplication Benchmark (CUDA)",
    marker=(:circle, 3),
    lw=2)
