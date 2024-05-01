import Pkg
Pkg.activate(".")  # environment in this folder
Pkg.add("Plots")
Pkg.add("CPUTime")
#Pkg.add( "BLIS" )
Pkg.add("MKL")
#Pkg.precompile()

using CPUTime
using Plots

using LinearAlgebra, MKL

function matrix_initialization(N)
    A = rand(Float32, N, N)
    B = rand(Float32, N, N)

    return A, B
end

function matrix_multiplication(A, B)
    return A * B
end

function parallel_multiply(A, B, N_threads)
    BLAS.set_num_threads(1)
    Threads.@threads for i in 1:N_threads
        matrix_multiplication(A, B)
    end
end

function serial_multiply(A, B, N_threads)
    BLAS.set_num_threads(N_threads)
    for i in 1:N_threads
        matrix_multiplication(A, B)
    end
end

# N_threads = Threads.nthreads()
N_threads = 8

A, B = matrix_initialization(100)

@time parallel_multiply(A, B, N_threads)
@time serial_multiply(A, B, N_threads)
