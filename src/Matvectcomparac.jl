using BenchmarkTools
using LinearAlgebra
using Plots

function matvec_mul_perf(n)
    A = rand(n, n)
    x = rand(n)
    

    result = @benchmark mul!($x, $A, $x)
    
    # Obtaining the time from the benchmark result
    time = minimum(result).time / 1e9  # Convert nanoseconds to seconds
    
    # Calculating GFLOPS
    flops = 2 * n^2
    gflops = flops / time
    
    return gflops
end

function matmat_mul_perf(n)
    A = rand(n, n)
    B = rand(n, n)
    C = similar(A)  # Create a new matrix for the result
    
    # Benchmarking the matrix-matrix multiplication
    result = @benchmark mul!($C, $A, $B)
    
    # Obtaining the time from the benchmark result
    time = minimum(result).time / 1e9  # Convert nanoseconds to seconds
    
    # Calculating GFLOPS
    flops = 2 * n^3
    gflops = flops / time
    
    return gflops
end

function plot_gflops_vs_matrix_size(max_size)
    sizes = 100:100:max_size
    matvec_gflops_data = Float64[]
    matmat_gflops_data = Float64[]
    
    for size in sizes
        push!(matvec_gflops_data, matvec_mul_perf(size))
        push!(matmat_gflops_data, matmat_mul_perf(size))
    end
    
    plot(sizes, matvec_gflops_data, xlabel="Matrix Size", ylabel="GFLOPS", label="Matrix-Vector Multiplication")
    plot!(sizes, matmat_gflops_data, label="Matrix-Matrix Multiplication")
end

plot_gflops_vs_matrix_size(2000)

