using LinearAlgebra
using MKL
using BenchmarkTools
using Plots

# Función para multiplicar una matriz NN por un vector N1, N veces
function matvec_mult(A, x)
    b = similar(x)
    for _ in 1:size(A, 1)
        b .= A * x
    end
    return b
end

# Función para medir GFLOPS
function gflops(A, x)
    N = size(A, 1)
    ops = 2 * N^3
    t = @belapsed matvec_mult($A, $x)
    gflops = (ops / t) / 1e9
    return gflops
end

# Configuración de tamaños de matriz
matrix_sizes = [128, 256, 512, 1024, 2048, 4096]
gflops_results = []

for N in matrix_sizes
    A = rand(N, N)
    x = rand(N)
    push!(gflops_results, gflops(A, x))
end

# Valor teórico de GFLOPS para una CPU Intel i7 10th gen
# Intel Core i7-10700K tiene 8 núcleos y 16 hilos, con un rendimiento de hasta 3.6 GHz.
# El rendimiento máximo teórico en GFLOPS es:
theoretical_gflops = 3.6 * 8 * 16

# Graficar resultados
plot(matrix_sizes, gflops_results, label="Measured GFLOPS", xlabel="Matrix Size (N)", ylabel="GFLOPS", lw=2, marker=:circle)
hline!([theoretical_gflops], label="Theoretical GFLOPS", lw=2, linestyle=:dash)
title!("GFLOPS vs Matrix Size")
#legend(:bottomright)

# Guardar gráfica
# savefig("gflops_vs_matrix_size.png")
