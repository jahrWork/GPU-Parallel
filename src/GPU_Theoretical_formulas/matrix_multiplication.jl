using CUDA
using BenchmarkTools
using Statistics
import LinearAlgebra.BLAS: vendor
using Plots

# Function that performs the matrix multiplication using CUDA
function gpu_matrix_multiply(A, B)
    CUDA.@sync A * B
end

function benchmark_matrix_multiplication(filename, n, num_runs)
    Time_GPU = zeros(num_runs)
    
    open(filename, "a") do file
        for i = 1:num_runs
            A = rand(Float32, n, n)
            B = rand(Float32, n, n)

            d_A = CUDA.CuArray(A)
            d_B = CUDA.CuArray(B)
            
            # Ensure the operation is compiled by running it once before timing
            gpu_matrix_multiply(d_A, d_B)

            # Benchmark the matrix multiplication function
            total_time_GPU = @belapsed gpu_matrix_multiply($d_A, $d_B) evals=1
            
            write(file, "$n, $(2 * n^3), $(vendor()), $total_time_GPU\n")
            
            Time_GPU[i] = total_time_GPU
        end
    end
    
    println("Matrix multiplication benchmarking for n = $n completed.")
    return Time_GPU
end

# Function to benchmark matrix multiplication for variable number of operations
function benchmark_matrix_multiplication_variable_ops(filename, n, N_ops)
    TIMES = max(1, div(N_ops, 2 * n^3))

    A = CUDA.rand(Float32, n, n)
    B = CUDA.rand(Float32, n, n)

    t = @elapsed for i = 1:TIMES
        CUDA.@sync A * B
    end
    println("Variable operations benchmarking for n = $n completed.")
    println("$n, $TIMES operations completed in $t seconds.")
    write(filename, "$n, $(TIMES * 2 * n^3), $t\n")
end

function benchmark_matrix_multiplication_fixed_runs(filename, n, num_runs)
    Time_GPU = zeros(num_runs)
    
    open(filename, "a") do file
        for i = 1:num_runs
            A = rand(Float32, n, n)
            B = rand(Float32, n, n)

            d_A = CUDA.CuArray(A)
            d_B = CUDA.CuArray(B)
            
            # Ensure the operation is compiled by running it once before timing
            gpu_matrix_multiply(d_A, d_B)

            # Benchmark the matrix multiplication function
            total_time_GPU = @belapsed gpu_matrix_multiply($d_A, $d_B) evals=1
            
            write(file, "$n, $(2 * n^3), $(vendor()), $total_time_GPU\n")
            
            Time_GPU[i] = total_time_GPU
        end
    end
    
    println("Fixed runs benchmarking for n = $n completed.")
    return Time_GPU
end


# Main function to perform benchmarks
function main()
    println("Starting benchmarking process...")
    filename = "benchmark_results_cuda.csv"
    N = 50:50:1450
    num_runs = 10
    Time_GPU_matrix_multiplication = zeros(length(N), num_runs)

    for (i, n) in enumerate(N)
        println("Benchmarking matrix multiplication for n = $n...")
        Time_GPU_matrix_multiplication[i, :] = benchmark_matrix_multiplication(filename, n, num_runs)
    end

    filename_fixed = "benchmark_results_cuda_fixed.csv"
    filename_variable = "datos_MM_NCTE_RAW.csv"
    open(filename_fixed, "w") do file
        write(file, "Matrix Size, Operations, BLAS Vendor, Duration (s)\n")
    end
    open(filename_variable, "w") do file
        write(file, "Matrix Size, Operations, Duration (s)\n")
    end

    N_fixed = 50:50:1450
    num_runs = 10
    N_variable = 50:25:2499
    N_ops_constant = 2 * 10000.0^3

    for n in N_fixed
        println("Benchmarking fixed runs for n = $n...")
        Time_GPU_fixed_runs = benchmark_matrix_multiplication_fixed_runs(filename_fixed, n, num_runs)
    end

    for n in N_variable
        println("Benchmarking variable operations for n = $n...")
        benchmark_matrix_multiplication_variable_ops(filename_variable, n, N_ops_constant)
    end

    plot(N, mean(Time_GPU_matrix_multiplication, dims=2), ribbon = std(Time_GPU_matrix_multiplication, dims=2), 
        xlabel="Matrix Size", ylabel="Duration (s)", title="Benchmark Matrix Multiplication (Fixed Runs)", legend=false)
    savefig("benchmark_results_fixed_runs.png")
    println("Benchmarking process completed.")
end

main()
