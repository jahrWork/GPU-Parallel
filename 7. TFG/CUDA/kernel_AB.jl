using CUDA
CUDA.allowscalar(false)
using BenchmarkTools

N = 50
Nt = 1000  # Número de iteraciones para las pruebas
a = CUDA.rand(N, N)
b = CUDA.rand(N, N)
c1 = CUDA.zeros(N, N)
c2 = CUDA.zeros(N, N)

# Kernel matmul1
function matmul1_kernel!(C, A, B, N)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x

    for i in idx:stride:N^2
        row = div(i - 1, N) + 1
        col = rem(i - 1, N) + 1
        temp = 0.0f0
        for k in 1:N
            temp += A[row, k] * B[k, col]
        end
        C[row, col] = temp
    end
    return
end

function matmul1_custom!(C::CuArray, A::CuArray, B::CuArray)
    N = size(A, 1)
    threads = 128
    blocks = cld(N^2, threads)
    @cuda threads=threads blocks=blocks matmul1_kernel!(C, A, B, N)
    return C
end

# Kernel matmul2
function matmul2_kernel!(C, A, B, N)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if i <= N && j <= N
        temp = 0.0f0
        for k in 1:N
            temp += A[i, k] * B[k, j]
        end
        C[i, j] = temp
    end
    return nothing
end

function matmul2_custom!(C::CuArray, A::CuArray, B::CuArray)
    N = size(A, 1)
    threads = (32, 32)
    blocks = (cld(N, threads[1]), cld(N, threads[2]))
    @cuda threads=threads blocks=blocks matmul2_kernel!(C, A, B, N)
    return C
end

# Pruebas de matmul1 y matmul2 con medición de tiempo
# Tiempo para matmul1
CUDA.synchronize()
elapsed1 = CUDA.@elapsed for _ in 1:Nt matmul1_custom!(c1, a, b) end
CUDA.synchronize()

# Tiempo para matmul2
CUDA.synchronize()
elapsed2 = CUDA.@elapsed for _ in 1:Nt matmul2_custom!(c2, a, b) end
CUDA.synchronize()

# Comparación de resultados
correct1 = isapprox(Array(c1), Array(a) * Array(b))
correct2 = isapprox(Array(c2), Array(a) * Array(b))

println("matmul1 es correcto: ", correct1, ", tiempo total: ", elapsed1, " segundos, tiempo promedio: ", elapsed1 / Nt, " segundos")
println("matmul2 es correcto: ", correct2, ", tiempo total: ", elapsed2, " segundos, tiempo promedio: ", elapsed2 / Nt, " segundos")
