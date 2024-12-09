using CUDA
CUDA.allowscalar(false)
using BenchmarkTools

N = 50
a = CUDA.rand(N, N)
b = CUDA.rand(N, N)
c1 = CUDA.zeros(N, N)
c2 = CUDA.zeros(N, N)

# Kernel matmul1
function matmul1_kernel!(C, A, B, N, Nt)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x

    for t in idx:stride:Nt
        @inbounds for i in 1:N
            for j in 1:N
                temp = 0.0f0
                for k in 1:N
                    temp += A[i, k] * B[k, j]
                end
                C[i, j] = temp
            end
        end
    end
    return
end

function matmul1_custom!(Nt, C::CuArray, A::CuArray, B::CuArray)
    N = size(A, 1)
    threads = 128
    blocks = cld(Nt, threads)
    @cuda threads=threads blocks=blocks matmul1_kernel!(C, A, B, N, Nt)
    return C
end

# Kernel matmul2
function matmul2_kernel!(C, A, B, N)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if i <= N && j <= N
        sum = zero(eltype(C))
        for k = 1:N
            sum += A[i, k] * B[k, j]
        end
        C[i, j] = sum
    end
    return nothing
end

function matmul2_custom!(Nt, C::CuArray, A::CuArray, B::CuArray)
    N = size(A, 1)
    threads = (16, 16)
    blocks = (cld(N, threads[1]), cld(N, threads[2]))
    for _ in 1:Nt
        @cuda threads=threads blocks=blocks matmul2_kernel!(C, A, B, N)
    end
    return C
end

# Pruebas de matmul1 y matmul2 con medición de tiempo
Nt = 1000000  # Número de iteraciones para matmul1

# Tiempo para matmul1
CUDA.synchronize()  # Sincroniza antes de medir
elapsed1 = CUDA.@elapsed matmul1_custom!(Nt, c1, a, b)
CUDA.synchronize()  # Asegura que termina antes de continuar

# Tiempo para matmul2
CUDA.synchronize()
elapsed2 = CUDA.@elapsed matmul2_custom!(Nt, c2, a, b)
CUDA.synchronize()

# Comparación de resultados
correct1 = isapprox(Array(c1), Array(a) * Array(b))
correct2 = isapprox(Array(c2), Array(a) * Array(b))

println("matmul1 es correcto: ", correct1, ", tiempo: ", elapsed1, " segundos")
println("matmul2 es correcto: ", correct2, ", tiempo: ", elapsed2, " segundos")
