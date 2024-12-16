using CUDA
CUDA.allowscalar(false)
using BenchmarkTools
using LinearAlgebra

N = 2050
a = CUDA.rand(Float32, N)
b = CUDA.rand(Float32, N)
# Changed c to a CuArray with 1 element because dot is a single number
c = CUDA.zeros(Float32, 1)

typeof(a) <: AbstractArray

function my_dot_kernel!(dot, a, b)
    i = CUDA.threadIdx().x + (CUDA.blockIdx().x - 1) * CUDA.blockDim().x
    if i <= length(a)
        # https://cuda.juliagpu.org/stable/development/kernel/#Atomics
        
        # Atomically add the product of a[i] * b[i] to the first (and only) element of 'dot'
        CUDA.@atomic dot[1] += a[i] * b[i]  # Atomic addition
    end
    return nothing
end

function my_dot_custom!(c::CuArray, a::CuArray, b::CuArray)
    @cuda blocks=cld(length(a), 1024) threads=1024 my_dot_kernel!(c, a, b)
    return nothing
end

my_dot_custom!(c, a, b)

# Check if the addition is correct
isapprox(Array(c)[1], dot(Array(a), Array(b)))