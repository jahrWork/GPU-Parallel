
using LinearAlgebra
using BenchmarkTools
using Plots

# Función para realizar la multiplicación de matriz por vector usando BLAS.gemv!
function matvec_mkl(A::Array{Float64, 2}, x::Array{Float64, 1})
    m, n = size(A)
    if length(x) != n
        throw(DimensionMismatch("Matrix columns must match vector length"))
    end

    y = zeros(Float64, m)  # El vector resultante

    alpha = 1.0  # Escalar para multiplicar
    beta = 0.0   # Escalar para el vector y (C := alpha*A*x + beta*y)

    # Llamada a la función BLAS.gemv!
    BLAS.gemv!('N', alpha, A, x, beta, y)

    return y
end

# Función para realizar el benchmark
function benchmark_matvec()
    sizes = [100, 200, 500, 1000, 2000, 5000, 10000]
    times = []

    for size in sizes
        A = rand(Float64, size, size)
        x = rand(Float64, size)
        println("Benchmarking for size $size")
        result = @benchmark matvec_mkl($A, $x)
        push!(times, minimum(result).time)
    end

    return sizes, times
end

sizes, times = benchmark_matvec()

# Graficar los resultados
plot(sizes, times, title="Benchmark of Matrix-Vector Multiplication",
     xlabel="Matrix Size (N x N)", ylabel="Time (ns)", legend=false, lw=2, marker=:o)

