import Pkg


using MKL
using BenchmarkTools
using Plots

# Función para realizar la multiplicación de matriz por vector usando MKL
function matvec_mkl(A::Array{Float64, 2}, x::Array{Float64, 1})
    y = zeros(Float64, size(A, 1))
    trans = Ref('N' % UInt8)  # Convertir a UInt8
    m, n = size(A)
    alpha = 1.0
    beta = 0.0
    lda = m
    incx = 1
    incy = 1
    ccall((:cblas_dgemv, MKL.libmkl_rt), Cvoid,
          (Ref{UInt8}, Ptr{Int32}, Ptr{Int32}, Ptr{Float64}, Ptr{Float64}, Ptr{Int32}, Ptr{Float64}, Ptr{Int32}, Ptr{Float64}, Ptr{Float64}, Ptr{Int32}),
          trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
    return y
end

# Función para realizar la multiplicación de matriz por matriz usando MKL
function matmat_mkl(A::Array{Float64, 2}, B::Array{Float64, 2})
    C = zeros(Float64, size(A, 1), size(B, 2))
    transa = Ref('N' % UInt8)  # Convertir a UInt8
    transb = Ref('N' % UInt8)  # Convertir a UInt8
    m, n = size(A)
    k = size(B, 1)
    alpha = 1.0
    beta = 0.0
    lda = m
    ldb = k
    ldc = m
    ccall((:cblas_dgemm, MKL.libmkl_rt), Cvoid,
          (Ref{UInt8}, Ref{UInt8}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Float64}, Ptr{Float64}, Ptr{Int32}, Ptr{Float64}, Ptr{Int32}, Ptr{Float64}, Ptr{Float64}, Ptr{Int32}),
          transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    return C
end

# Función para realizar el benchmark
function benchmark_operations()
    sizes = [100, 200, 500, 1000, 2000, 5000, 10000]
    times_vec = Float64[]
    times_mat = Float64[]

    for size in sizes
        A = rand(Float64, size, size)
        x = rand(Float64, size)
        B = rand(Float64, size, size)
        
        println("Benchmarking for size $size (Matrix-Vector)")
        result_vec = @benchmark matvec_mkl($A, $x)
        push!(times_vec, median(result_vec).time)
        
        println("Benchmarking for size $size (Matrix-Matrix)")
        result_mat = @benchmark matmat_mkl($A, $B)
        push!(times_mat, median(result_mat).time)
    end

    return sizes, times_vec, times_mat
end

sizes, times_vec, times_mat = benchmark_operations()

# Graficar los resultados
plot(sizes, times_vec, title="Benchmark of Matrix Operations",
     xlabel="Matrix Size (N x N)", ylabel="Time (ns)", label="Matrix-Vector (MKL)",
     lw=2, marker=:o)
plot!(sizes, times_mat, label="Matrix-Matrix (MKL)", lw=2, marker=:x)

