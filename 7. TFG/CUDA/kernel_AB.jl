using CUDA
CUDA.allowscalar(false)
using BenchmarkTools

N = 50
Nt = 100000  # Número de iteraciones para las pruebas
a = CUDA.rand(N, N)
b = CUDA.rand(N, N)
c1 = CUDA.zeros(N, N)
c2 = CUDA.zeros(N, N)
c3 = CUDA.zeros(N, N)

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

# Kernel matmul3
function matmul3_kernel!(C, A, B, N, Nt)
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

function matmul3_custom!(C::CuArray, A::CuArray, B::CuArray, Nt)
    N = size(A, 1)
    threads = 128
    blocks = cld(Nt, threads)
    @cuda threads=threads blocks=blocks matmul3_kernel!(C, A, B, N, Nt)
    return C
end

# Pruebas de matmul1, matmul2 y matmul3 con medición de tiempo

# Tiempo para matmul1
CUDA.synchronize()
elapsed1 = CUDA.@elapsed for _ in 1:Nt matmul1_custom!(c1, a, b) end
CUDA.synchronize()

# Tiempo para matmul2
CUDA.synchronize()
elapsed2 = CUDA.@elapsed for _ in 1:Nt matmul2_custom!(c2, a, b) end 
CUDA.synchronize()

# Tiempo para matmul3
CUDA.synchronize()
elapsed3 = CUDA.@elapsed matmul3_custom!(c3, a, b, Nt) # Nt ya se incluye en el kernel
CUDA.synchronize()

# Comparación de resultados
correct1 = isapprox(Array(c1), Array(a) * Array(b))
correct2 = isapprox(Array(c2), Array(a) * Array(b))
correct3 = isapprox(Array(c3), Array(a) * Array(b))

println("matmul1 es correcto: ", correct1, ", tiempo total: ", elapsed1, " segundos, tiempo promedio: ", elapsed1 / Nt, " segundos")
println("matmul2 es correcto: ", correct2, ", tiempo total: ", elapsed2, " segundos, tiempo promedio: ", elapsed2 / Nt, " segundos")
println("matmul3 es correcto: ", correct3, ", tiempo total: ", elapsed3, " segundos, tiempo promedio: ", elapsed3 / Nt, " segundos")
