using LinearAlgebra
using MKL
using BenchmarkTools
using Plots

A = rand(1000, 1000)
x = rand(1000)

b = Vector{Float64}(undef, size(A, 1))

BLAS.gemv!('N', 1.0, A, x, 0.0, b)

println("b = $b")

function gflops(A, x, N)
    ops = 2 * N^3
    t = @belapsed BLAS.gemv!('N', 1.0, $A, $x, 0.0, $b)
    gflops = (ops / t) / 1e9
    return gflops
end

matrix_sizes = [128, 256, 512, 1024, 2048, 4096,10000]
gflops_results = []

for N in matrix_sizes
    A = rand(N, N)
    x = rand(N)
    b = Vector{Float64}(undef, size(A, 1))
    println("N = $N")
    push!(gflops_results, gflops(A, x, N))
    println("GFLOPS = $(gflops(A, x, N))")
end

N_Cores = 6
theoretical_time = 1e-9 / (4e9 * (512/32) * N_Cores)
theoretical_gflops = 1 / theoretical_time

plot(matrix_sizes, gflops_results, label="Measured GFLOPS", xlabel="Matrix Size (N)", ylabel="GFLOPS", lw=2, marker=:circle)
hline!([theoretical_gflops], label="Theoretical GFLOPS", lw=2, linestyle=:dash)
title!("GFLOPS vs Matrix Size")