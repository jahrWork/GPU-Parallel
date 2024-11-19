using CPUTime, Plots, LinearAlgebra, MKL, CpuId, BenchmarkTools

# Define la función get_avx_value
function get_avx_value(string_cpuid)
    # Inicializar la variable AVX_value
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

# Define la función init para inicializar matrices y operaciones (solo MxM)
function init(problem, N)
    A = rand(Float32, N, N)
    B = rand(Float32, N, N)
    Nop = 2 * N^3
    return A, B, Nop
end

# Función original para A * B (solo MxM)
function mult(problem, A, B)
    return A * B
end

# Función mul! (solo MxM)
function mult!(problem, A, B, C)
    mul!(C, A, B)
end

# Función para comparar el rendimiento de A * B y mul! (solo MxM)
function compare_multiplication_methods(problem, N, N_cores)
    Time_A_mul_B = zeros(length(N))
    Time_mul_B = zeros(length(N))
    println("Problem = ", problem)

    for (i, n) in enumerate(N)
        A, B, Nop = init(problem, n)
        C = similar(B)  # create an array to store the result in-place for mul!

        # Benchmark A * B (original)
        # The $ symbol is used to interpolate the variables problem, A, and B into the @belapsed macro,
        # replacing them with their values before running, which makes the benchmarking more accurate.
        dt1 = @belapsed mult($problem, $A, $B)
        Time_A_mul_B[i] = (dt1 / Nop) * 1e9  # Convert seconds to nanoseconds

        # Benchmark mul! (in-place)
        dt2 = @belapsed mult!($problem, $A, $B, $C)
        Time_mul_B[i] = (dt2 / Nop) * 1e9  # Convert seconds to nanoseconds

        println("N=", n, " Time per operation (A * B) = ", Time_A_mul_B[i], " ns")
        println("N=", n, " Time per operation (mul!) = ", Time_mul_B[i], " ns")
    end

    return Time_A_mul_B, Time_mul_B
end

# Función para graficar y comparar GFLOPS de A * B y mul!
function plot_GFLOPS_comparison()
    # CPU Features and configuration
    cpuid = cpuinfo()
    string_cpuid = string(cpuid)
    AVX_value = get_avx_value(string_cpuid)
    N_threads = Threads.nthreads()
    N_cores = N_threads / 2

    # Range of matrix dimensions to test
    N = Vector([25:25:500; 550:50:1000])

    # Set the number of BLAS threads based on the number of cores
    BLAS.set_num_threads(N_threads)

    # Perform timings with both A * B and mul!
    Time1_A_mul_B, Time1_mul_B = compare_multiplication_methods("MxM", N, N_cores)

    # Calculate GFLOPS
    GFLOPS1_A_mul_B = 1 ./ Time1_A_mul_B
    GFLOPS1_mul_B = 1 ./ Time1_mul_B
    
    # Plot comparison for Mat x Mat
    plot(N, GFLOPS1_A_mul_B, title = "GFLOPS Comparison - Mat x Mat", xlabel = "\$ N \$", ylabel = "GFLOPS", label = "A * B", lw = 3, xlimits = (0, 1000), ylimits = (0, maximum(GFLOPS1_A_mul_B) + 100))
    plot!(N, GFLOPS1_mul_B, label = "mul!", lw = 3)
    savefig("plot_GFLOPS_comparison_mul!_MxM.png")
end

# Ejecuta la comparación de GFLOPS
plot_GFLOPS_comparison()
