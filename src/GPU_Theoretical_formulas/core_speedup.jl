import Pkg
Pkg.activate(".")
using LinearAlgebra, MKL
 
N = 10_000
A = rand(Float32, N, N)
B = rand(Float32, N, N)
 
function matrix_multiplication(A, B)
    return A * B
end
 
N_threads = [1, 2, 4, 8, 16, 32]
 
matrix_multiplication(A, B)
 
for threads in N_threads
    BLAS.set_num_threads(threads)
    @time matrix_multiplication(A, B)
end