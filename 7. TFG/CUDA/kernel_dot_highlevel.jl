using CUDA
CUDA.allowscalar(false)
using BenchmarkTools
using LinearAlgebra
CUDA.@profile external=true

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
    CUDA.synchronize()
    return nothing
end

println("High-level custom dot kernel started running")

CUDA.synchronize()

my_dot_custom!(c, a, b)

# Check if the addition is correct
println("isapprox: ", isapprox(Array(c)[1], dot(Array(a), Array(b))))

# Benchmark de dot oficial
official_dot_benchmark = @benchmark dot($a, $b) samples=1000 evals=100 seconds=30
official_dot_mean = mean(official_dot_benchmark.times) / 1e9 # Convertir de nanosegundos a segundos

# Benchmark de dot personalizado
custom_dot_benchmark = @benchmark my_dot_custom!($c, $a, $b) samples=1000 evals=100 seconds=30
custom_dot_mean = mean(custom_dot_benchmark.times) / 1e9 # Convertir de nanosegundos a segundos

println("Tiempo oficial dot: ", official_dot_mean, " segundos")
println("Tiempo dot high-level: ", custom_dot_mean, " segundos")
println(" ")
println("Speedup: ", official_dot_mean / custom_dot_mean)
