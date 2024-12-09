using CUDA
CUDA.allowscalar(false)
using BenchmarkTools

N = 2050
a = CUDA.rand(N)
b = CUDA.rand(N)
c = similar(a)

typeof(a) <: AbstractArray

function my_add!(c::AbstractArray, a::AbstractArray, b::AbstractArray)
    for i in eachindex(c)
        c[i] = a[i] + b[i]
    end
    nothing
end

function my_add_kernel!(c, a, b)
    i = CUDA.threadIdx().x + (blockIdx().x - 1) * blockDim().x


    # Importante este if, porque si no, se intenta acceder a elementos de a y b que no existen.
    # Ver vídeo, minuto 20:30. Intentaría coger elementos vacios del bloque que quede al final
    # si no acaba justo completando el bloque.
    if i <= length(c)
        c[i] = a[i] + b[i]
    end
    return nothing
end

function my_add_custom!(c::CuArray, a::CuArray, b::CuArray)
    @cuda blocks=cld(length(c), 1024) threads=1024 my_add_kernel!(c, a, b)
end

my_add_custom!(c, a, b)

# Check if the addition is correct
isapprox(Array(c), Array(a) .+ Array(b))
