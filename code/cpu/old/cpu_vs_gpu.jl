using LinearAlgebra
using CUDA
using BenchmarkTools
using Plots  # Para graficar

# Función para calcular GFLOPS
function gflops_mul(A, B, C, time_secs)
    n = size(A, 1)
    return (2 * n^3) / (time_secs * 1e9)  # GFLOPS
end

# Benchmark para BLAS (CPU)
function benchmark_blas(N_range)
    println(" ")
    println("BLAS (CPU) Benchmark:")
    println(" ")
    gflops_results = []
    memory_time_results = []
    for N in N_range
        A = rand(N, N)
        B = rand(N, N)
        C = zeros(N, N)
        
        # Warm up
        C .= A * B

        println("Tamaño: $N x $N")

        # Time the matrix multiplication
        # @btime mul!($C, $A, $B) # Julia ejecuta esta operación múltiples veces internamente para medir de manera precisa el tiempo mínimo

        # Measure the performance
        time = @belapsed mul!($C, $A, $B)
        gflops = gflops_mul(A, B, C, time)
        push!(gflops_results, gflops)

        # Medir el tiempo de asignación de memoria
        memory_time = @belapsed begin
            A = rand($N, $N)  # Corregir aquí usando $
            B = rand($N, $N)  # Corregir aquí usando $
            C = zeros($N, $N)  # Corregir aquí usando $
        end
        push!(memory_time_results, memory_time * 1e3)  # Convertir a milisegundos
        println("  GFLOPS: $gflops, Asignación de Memoria: $(memory_time * 1e3) ms")
        println(" ")
    end
    return gflops_results, memory_time_results
end

# Benchmark para CUDA (GPU)
function benchmark_cuda(N_range)
    println(" ")
    println("CUDA (GPU) Benchmark:")
    println(" ")
    gflops_results = []
    memory_time_results = []
    for N in N_range
        A_gpu = CUDA.rand(Float32, N, N)
        B_gpu = CUDA.rand(Float32, N, N)
        C_gpu = CUDA.zeros(Float32, N, N)  # Inicializar C_gpu antes de los bloques de tiempo

        # Warm up
        C_gpu .= A_gpu * B_gpu

        println("Tamaño: $N x $N")

        # Time the matrix multiplication
        # @btime CUDA.@sync $C_gpu .= $A_gpu * $B_gpu # Julia ejecuta esta operación múltiples veces internamente para medir de manera precisa el tiempo mínimo

        # Medir el rendimiento
        time = @belapsed CUDA.@sync $C_gpu .= $A_gpu * $B_gpu
        gflops = gflops_mul(A_gpu, B_gpu, C_gpu, time)
        push!(gflops_results, gflops)

        # Medir el tiempo de asignación de memoria
        memory_time = @belapsed begin
            A_gpu = CUDA.rand(Float32, $N, $N)  # Corregir aquí usando $
            B_gpu = CUDA.rand(Float32, $N, $N)  # Corregir aquí usando $
            C_gpu = CUDA.zeros(Float32, $N, $N)  # Corregir aquí usando $
        end
        push!(memory_time_results, memory_time * 1e3)  # Convertir a milisegundos
        println("  GFLOPS: $gflops, Asignación de Memoria: $(memory_time * 1e3) ms")
        println(" ")
    end
    return gflops_results, memory_time_results
end

# Rango de tamaños de matrices
N_range = 200:200:2000

# Ejecutar benchmarks
blas_gflops, blas_memory_times = benchmark_blas(N_range)
cuda_gflops, cuda_memory_times = benchmark_cuda(N_range)

# Crear gráfico de GFLOPS en el eje Y izquierdo
p1 = plot(N_range, blas_gflops, label="BLAS (CPU) GFLOPS", marker=:circle, linewidth=2, color=:blue, ylim=(0, maximum(blas_gflops) * 1.1))  # Establecer un límite superior basado en los datos
plot!(p1, N_range, cuda_gflops, label="CUDA (GPU) GFLOPS", marker=:diamond, linewidth=2, color=:green, ylim=(0, maximum(cuda_gflops) * 1.1))
ylabel!("GFLOPS")
xlabel!("Tamaño de la Matriz (N)")
title!("Comparación de GFLOPS y Tiempos de Asignación de Memoria: BLAS vs CUDA")

# Crear el segundo eje Y para los tiempos de memoria (ms)
p2 = twinx(p1)  # Crear un segundo eje Y en la misma gráfica
plot!(p2, N_range, blas_memory_times, label="BLAS (CPU) Memoria (ms)", marker=:square, linewidth=2, color=:red, linestyle=:dash, ylim=(0, maximum(blas_memory_times) * 1.1))  # Establecer límite superior
plot!(p2, N_range, cuda_memory_times, label="CUDA (GPU) Memoria (ms)", marker=:x, linewidth=2, color=:orange, linestyle=:dash, ylim=(0, maximum(cuda_memory_times) * 1.1))
ylabel!(p2, "Tiempo de Asignación de Memoria (ms)")

# Combinar las leyendas de ambas gráficas
plot!(p1, legend=:topleft)
plot!(p2, legend=:topright)

# Guardar la gráfica en un archivo PNG
savefig("blas_vs_cuda_200_2000.png")

# Rango de tamaños de matrices
N_range = 2000:1000:10000

# Ejecutar benchmarks
blas_gflops, blas_memory_times = benchmark_blas(N_range)
cuda_gflops, cuda_memory_times = benchmark_cuda(N_range)

# Crear gráfico de GFLOPS en el eje Y izquierdo
p3 = plot(N_range, blas_gflops, label="BLAS (CPU) GFLOPS", marker=:circle, linewidth=2, color=:blue, ylim=(0, maximum(blas_gflops) * 1.1))  # Establecer un límite superior basado en los datos
plot!(p3, N_range, cuda_gflops, label="CUDA (GPU) GFLOPS", marker=:diamond, linewidth=2, color=:green, ylim=(0, maximum(cuda_gflops) * 1.1))
ylabel!("GFLOPS")
xlabel!("Tamaño de la Matriz (N)")
title!("Comparación de GFLOPS y Tiempos de Asignación de Memoria: BLAS vs CUDA")

# Crear el segundo eje Y para los tiempos de memoria (ms)
p4 = twinx(p3)  # Crear un segundo eje Y en la misma gráfica
plot!(p4, N_range, blas_memory_times, label="BLAS (CPU) Memoria (ms)", marker=:square, linewidth=2, color=:red, linestyle=:dash, ylim=(0, maximum(blas_memory_times) * 1.1))  # Establecer límite superior
plot!(p4, N_range, cuda_memory_times, label="CUDA (GPU) Memoria (ms)", marker=:x, linewidth=2, color=:orange, linestyle=:dash, ylim=(0, maximum(cuda_memory_times) * 1.1))
ylabel!(p4, "Tiempo de Asignación de Memoria (ms)")

# Combinar las leyendas de ambas gráficas
plot!(p3, legend=:topleft)
plot!(p4, legend=:topright)

# Guardar la gráfica en un archivo PNG
savefig("blas_vs_cuda_2000_10000.png")
