using LinearAlgebra
using MKL
using BenchmarkTools
using Plots


function matvec_mult(A, x, N)
    b = zeros(size(x)) 
    for _ in 1:N
        b = A * x
    end
    return b
end

# Función para medir GFLOPS
function gflops(A, x, N)
    ops = 2 * N^3
    t = @belapsed matvec_mult($A, $x, $N)
    gflops = (ops / t) / 1e9
    return gflops
end

# Configuración de tamaños de matriz
matrix_sizes = [128, 256, 512, 1024, 2048, 4096,10000]
gflops_results = []

for N in matrix_sizes
    A = rand(N, N)
    x = rand(N)
    push!(gflops_results, gflops(A, x, N))
end


theoretical_gflops = 2,4 * 8 * 16


plot(matrix_sizes, gflops_results, label="Measured GFLOPS", xlabel="Matrix Size (N)", ylabel="GFLOPS", lw=2, marker=:circle)
hline!([theoretical_gflops], label="Theoretical GFLOPS", lw=2, linestyle=:dash)
title!("GFLOPS vs Matrix Size")

