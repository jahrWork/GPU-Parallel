using LinearAlgebra, BenchmarkTools, CpuId, Plots

# Función para obtener el soporte AVX de la CPU
function get_avx_value()
    AVX_value = 0

    if CpuId.cpufeature(:AVX512F)
        AVX_value = 16  # AVX-512
    elseif CpuId.cpufeature(:AVX2)
        AVX_value = 8   # AVX2
    elseif CpuId.cpufeature(:AVX)
        AVX_value = 8   # AVX
    elseif CpuId.cpufeature(:SSE)
        AVX_value = 4   # SSE
    else
        AVX_value = 0   # No SIMD support
    end

    return AVX_value
end

# Función para realizar el benchmark con matrices grandes y muchas iteraciones
function time_MxM_benchmark(N_range, num_iterations)
    GFLOPS = zeros(length(N_range))
    for (i, N) in enumerate(N_range)
        A = rand(Float32, N, N)
        B = rand(Float32, N, N)
        Nop = 2 * N^3  # Número de operaciones de punto flotante

        # Realizar múltiples multiplicaciones y calcular el tiempo promedio
        times = []
        for _ in 1:num_iterations
            GC.gc()  # Limpia la memoria antes de empezar
            t = @benchmarkable $A * $B
            result = run(t)
            push!(times, minimum(result.times))
        end
        GC.gc()  # Limpia la memoria después de la operación


        # Calcular el tiempo promedio
        avg_time_ns = mean(times)

        # Calcular GFLOPS
        time_s = avg_time_ns / 1e9  # Convertir nanosegundos a segundos
        GFLOPS[i] = (Nop / time_s) / 1e9  # Convertir a GFLOPS

        println("N = $N, Tiempo promedio = $(avg_time_ns) ns, GFLOPS = $(GFLOPS[i])")
    end
    return GFLOPS
end

# Función principal para realizar el benchmark y maximizar el uso de CPU
function plot_GFLOPS_MxM()
    # Configurar BLAS para usar todos los hilos lógicos
    num_threads = Sys.CPU_THREADS
    BLAS.set_num_threads(12)
    println("Hilos BLAS establecidos en ", BLAS.get_num_threads())

    # Obtener el valor AVX
    AVX_value = get_avx_value()
    println("Valor AVX detectado: ", AVX_value)

    if AVX_value == 0
        println("No se detectó soporte AVX adecuado. Saliendo.")
        return
    end

    # Rango de N aumentado para generar más carga de trabajo
    N_range = collect(30:10:200)  # Incrementa los valores de N para forzar mayor carga

    # Imprimir información sobre el proveedor de BLAS
    println("Proveedor BLAS: ", BLAS.vendor())
    

    println(" ")

    # Medir GFLOPS con múltiples iteraciones para mayor precisión
    GFLOPS = time_MxM_benchmark(N_range, 1)  # 3 iteraciones por N

    println(" ")

    # Obtener la frecuencia de la CPU (solo funciona en procesadores Intel creo)
    CPU_frequency_GHz = cpu_base_frequency()
    if CPU_frequency_GHz == 0.0
        println("No se pudo detectar la frecuencia de la CPU automáticamente.")
        println("Por favor, introduce la frecuencia de tu CPU en GHz (por ejemplo, 3.9): ")
        CPU_frequency_GHz = parse(Float64, readline())
    end
    CPU_frequency = CPU_frequency_GHz * 1e9  # Convertir a Hz
    N_logical_cores = num_threads

    Theoretical_GFLOPS = (CPU_frequency * AVX_value * 2 * N_logical_cores) / 1e9
    println("Frecuencia de CPU utilizada: ", CPU_frequency_GHz, " GHz")
    println("GFLOPS teóricos (incluyendo hilos lógicos) = ", Theoretical_GFLOPS)

    # Graficar los resultados
    plot(N_range, GFLOPS;
         title = "GFLOPS vs N (Usando 100% de la CPU)",
         xlabel = "N",
         ylabel = "GFLOPS",
         label = "GFLOPS Medidos",
         lw = 3,
         legend=:bottomright)
    hline!([Theoretical_GFLOPS], label = "GFLOPS Teóricos", linestyle=:dash)
    display(current())
end

plot_GFLOPS_MxM()
