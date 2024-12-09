using LinearAlgebra, CUDA
using BenchmarkTools
using Printf

include("system_info.jl")  # Incluye información del sistema para obtener GFLOPS teóricos

# Operación de multiplicación de matrices en GPU
function matmul_gpu(Nt, A, B, C)
    for j in 1:Nt
        C .= A * B
    end
    return C
end

function mul_gpu(Nt, A, B, C)
    for j in 1:Nt
        mul!(C, A, B)
    end
    return C
end

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

function matmul1_gpu!(Nt, A::CuArray, B::CuArray, C::CuArray)
    N = size(A, 1)
    threads = 128
    blocks = cld(N^2, threads)

    for _ in 1:Nt
        @cuda threads=threads blocks=blocks matmul1_kernel!(C, A, B, N)
    end
    return C
end

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
    return
end

function matmul2_gpu!(Nt, C::CuArray, A::CuArray, B::CuArray)
    N = size(A, 1)
    threads = (16, 16)
    blocks = (cld(N, threads[1]), cld(N, threads[2]))

    for _ in 1:Nt
        @cuda threads=threads blocks=blocks matmul2_kernel!(C, A, B, N)
    end
    return C
end

# Benchmarking sin uso de hilos de CPU
function measure_gpu(operations, Nt, N, Nop)
    # Crear matrices en GPU
    A = CUDA.randn(Float32, N, N)
    B = CUDA.randn(Float32, N, N)
    C = CUDA.zeros(Float32, N, N)

    # Warm-up
    operations(Nt, A, B, C)

    # Medir tiempo
    CUDA.synchronize()
    dt_cu_elapsed = CUDA.@elapsed operations(Nt, A, B, C)
    CUDA.synchronize()

    # Calcular GFLOPS
    Time_cu_elapsed = (dt_cu_elapsed / Nop) * 1e9  # Convertir segundos a nanosegundos
    GFLOPS_cu_elapsed = round(1 / Time_cu_elapsed, digits=2)

    pretty_print(N, Nt, operations, GFLOPS_cu_elapsed, "GPU")
    return GFLOPS_cu_elapsed
end

# Función para imprimir resultados de forma ordenada
function pretty_print(c1, c2, c3, c4, c5)
    @printf("%10s %10s %20s %15s  %10s\n", c1, c2, c3, c4, c5)
end

# Ejecución de pruebas
println("Pretty_Print GPU GFLOPS")
println(" ")

# Obtener GFLOPS máximos teóricos
max_gpu_gflops = round(gpu_info(true), digits=2)
println("Theoretical max GPU GFLOPS: ", max_gpu_gflops, " GFLOPS")
println(" ")

pretty_print("N", "Nt", "Operations", "GFLOPS", "Device")
dims = [(50, 100000), (100, 10000), (200, 2000), (400, 1000), (800, 100)]
test = [matmul_gpu, mul_gpu, matmul1_gpu!, matmul2_gpu!]

for (N, Nt) in dims
    for f in test
        GFLOPS = measure_gpu(f, Nt, N, 2 * N^3 * Nt)
    end
    println(" ")
end
