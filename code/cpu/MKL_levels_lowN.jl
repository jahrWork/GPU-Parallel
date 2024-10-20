# Importar paquetes necesarios
using CPUTime, Plots, LinearAlgebra, MKL, CpuId, Statistics
gr()  # Asegurarse de usar el backend GR

# Función para obtener el valor de AVX
function get_avx_value(string_cpuid)
    if occursin("256 bit", string_cpuid)
        return 8
    elseif occursin("512 bit", string_cpuid)
        return 16
    else
        return 0
    end
end

# Función para inicializar matrices para MxM
function init_MxM(N)
    A = rand(Float32, N, N)
    B = rand(Float32, N, N)
    Nop = 2 * N^3
    return A, B, Nop
end

# Función para multiplicar matrices (MxM)
function mult_MxM(A, B)
    return A * B
end

# Función para medir el tiempo de multiplicación MxM con repeticiones
function time_multiplication_MxM(N, repetitions)
    A, B, Nop = init_MxM(N)
    
    times = Float64[]
    
    @inbounds @simd for _ in 1:repetitions
        t1 = time_ns()
        mult_MxM(A, B)
        t2 = time_ns()
        dt = t2 - t1
        push!(times, dt / Nop)  # Guardar el tiempo por operación
    end
    
    return mean(times)  # Devolver el promedio de las repeticiones
end

# Función principal para plotear GFLOPS para MxM con repeticiones
function plot_GFLOPS_MxM(repetitions = 5)
    # Obtener información de la CPU
    cpuid = cpuinfo()
    string_cpuid = string(cpuid)
    println("Soporte AVX: ", occursin("256", string_cpuid))
    println("Soporte AVX-512: ", occursin("512 bit", string_cpuid))
    
    AVX_value = get_avx_value(string_cpuid)
    
    # Configurar el número de hilos
    N_threads = 12
    BLAS.set_num_threads(12)
    println("Número de hilos BLAS configurados: ", BLAS.get_num_threads())
    
    # Definir el rango de dimensiones de matrices a probar
    N = 100:100:2000  # Puedes ajustar este rango según tus necesidades
    
    # Calcular el tiempo teórico por operación
    # Suponiendo una frecuencia de 3.9 GHz y uso completo de AVX
    Theoretical_time = 1e9 / (3.9e9 * AVX_value * 2 * N_threads)
    println("Tiempo teórico por operación: ", Theoretical_time, " nsec")
    
    # Inicializar vector para almacenar los tiempos
    Time = Float64[]
    
    # Medir el tiempo para cada N con repeticiones
    for n in N
        dt_per_op = time_multiplication_MxM(n, repetitions)
        push!(Time, dt_per_op)
        println("N = ", n, " | Tiempo por operación (promedio de $repetitions repeticiones) = ", dt_per_op, " nsec")
    end
    
    # Calcular GFLOPS
    GFLOPS = 1 ./ Time
    GFLOPS_max = 1 / Theoretical_time
    
    println("GFLOPS máximo teórico: ", GFLOPS_max)
    
    # Datos para graficar
    x = Float64.(N)
    y_theoretical = fill(GFLOPS_max, length(GFLOPS))
    
    # Graficar los resultados
    plot(x, GFLOPS, 
         title = "GFLOPS vs Dimensión de la Matriz (MxM) con MKL",
         xlabel = "Dimensión N",
         ylabel = "GFLOPS",
         label = "MxM",
         lw = 2,
         xlim = (minimum(x), maximum(x)),
         ylim = (0, max(GFLOPS_max, maximum(GFLOPS)) * 1.1))
    
    plot!(x, y_theoretical, 
          label = "Teórico", 
          lw = 2, 
          linestyle = :dash)
    
    savefig("GFLOPS_vs_N.png")  # Guardar el gráfico como archivo de imagen
    display(plot)  # Asegurar que el gráfico se muestre
end

# Ejecutar la función de plot con repeticiones
plot_GFLOPS_MxM(50000)  # Puedes cambiar el número de repeticiones si lo deseas
