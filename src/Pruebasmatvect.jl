using BenchmarkTools
using LinearAlgebra
using Plots

function matvec_mul_perf(n)
    A = rand(n, n)
    x = rand(n)
    
    # Benchmarking the matrix-vector multiplication
    time = @belapsed mul!($x, $A, $x)
    
    # Calculating GFLOPS
    flops = 2 * n^2
    gflops = flops / (time * 1e9)
    
    return gflops
end

function plot_gflops_vs_matrix_size(max_size)
    sizes = 100:100:max_size
    gflops_data = Float64[]
    
    for size in sizes
        push!(gflops_data, matvec_mul_perf(size))
    end
    
    plot(sizes, gflops_data, xlabel="Matrix Size", ylabel="GFLOPS", label="", title="GFLOPS vs Matrix Size")
end

plot_gflops_vs_matrix_size(2000)
