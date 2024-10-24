using Base.Threads
using CPUTime, Plots, LinearAlgebra, MKL, CpuId, Statistics

function get_avx_value(string_cpuid)
    # Inicializar la variable AVX_Value
    AVX_value = 0

    # Buscar el tamaño del vector SIMD en la cadena y asignar el valor correspondiente
    if occursin("256 bit", string_cpuid)
        AVX_value = 8
    elseif occursin("512 bit", string_cpuid)
        AVX_value = 16
    else
        AVX_value = 0
    end
    
    return AVX_value
end

function init(problem, N) 
    if problem == "MxM" 
        A = rand(Float32, N, N )
        B = rand(Float32, N, N )
        Nop = 2 * N^3 
    elseif problem == "MxV"
        A = rand(Float32, N, N )
        B = rand(Float32, N, 1 )
        Nop = 2 * N^2 
    elseif problem == "VxV"
        A = rand(Float32, N, 1 )
        B = rand(Float32, N, 1 )
        Nop = 2 * N 
    end 

    return A, B, Nop 
end

function mult(problem, A, B)
    if problem == "MxM" 
        return A * B  
    elseif problem == "MxV"
        return A * B
    elseif problem == "VxV"
        return transpose(A) * B
    end 
end

function time_multiplication_parallel(problem, N, N_cores, repeats=100)
    Time = zeros(length(N))
    
    for i in 1:length(N)
        n = N[i]
        times = zeros(repeats)

        @threads for r in 1:repeats  # Distribuir las repeticiones entre los threads
            A, B, Nop = init(problem, n)
            t1 = time_ns()
            mult(problem, A, B)
            t2 = time_ns()
            dt = t2 - t1
            times[r] = dt / Nop
        end

        Time[i] = mean(times)  # Calcular el promedio de las repeticiones
        println("N=", n, " Average Time per operation =", Time[i], " nsec over ", repeats, " repeats")
    end
    
    return Time
end

function plot_GFLOPS_parallel()

    # CPU Features
    cpuid = cpuinfo()
    string_cpuid = string(cpuid)
    println("AVX support: ", occursin("256", string_cpuid))
    println("AVX-512 support: ", occursin("512 bit", string_cpuid))

    AVX_value = get_avx_value(string_cpuid)

    # Number of cores
    N_cores = 4

    # Range of matrix dimensions to test
    N = 10:10:200  # Ajustamos N de 10 a 200

    # Set the number of BLAS threads based on the number of cores
    BLAS.set_num_threads(2 * N_cores)
    println(" threads = ", BLAS.get_num_threads(), " N_cores =", N_cores)

    # Repetir múltiples veces en rangos bajos de N para mejorar precisión
    repeats = 1000  # Repetir muchas veces para promediar

    # Time the matrix multiplication and matrix-vector multiplication operations
    Theoretical_time = 1e9 / (4.5e9 * AVX_value * 2 * N_cores)
    println(" Theoretical time per operation =", Theoretical_time, " nsec")

    Time1 = time_multiplication_parallel("MxM", N, N_cores, repeats)
    Time2 = time_multiplication_parallel("MxV", N, N_cores, repeats)
    Time3 = time_multiplication_parallel("VxV", N, N_cores, repeats)

    # Calculate GFLOPS (floating-point operations per second)
    GFLOPS1 = 1 ./ Time1
    GFLOPS2 = 1 ./ Time2
    GFLOPS3 = 1 ./ Time3
    GFLOPS_max = 1 / Theoretical_time

    # Data for plotting
    x = float(N)
    GFLOPS4 = fill(GFLOPS_max, length(GFLOPS1))

    max1 = maximum(GFLOPS1)
    max2 = maximum(GFLOPS2)
    println("max GFLOPS Mat x Mat = ", max1)
    println("max GFLOPS Mat x Vect = ", max2)
    println("Ratio max1/ max2 = ", max1 / max2)

    # Primer plot para Mat x Mat
    plot(N, GFLOPS1,
         title="GFLOPS versus number of operations",
         xlabel="\$ N \$", ylabel="GFLOPS",
         label="Mat x Mat", lw=3,
         xlimits=(0, 200), ylimits=(0, 800)
    )

    # Añadir Mat x Vect
    plot!(N, GFLOPS2, label="Mat x Vect", lw=3)

    # Añadir Vect x Vect
    plot!(N, GFLOPS3, label="Vect x Vect", lw=3)

    # Añadir Theoretical
    plot!(N, GFLOPS4, label="Theoretical", lw=3)

end

plot_GFLOPS_parallel()
