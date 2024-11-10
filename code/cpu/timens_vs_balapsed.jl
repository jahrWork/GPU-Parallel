using CPUTime, Plots, LinearAlgebra, MKL, CpuId, BenchmarkTools

function get_avx_value(string_cpuid)
    AVX_value = 0
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
        A = rand(Float32, N, N)
        B = rand(Float32, N, N)
        Nop = 2 * N^3
    elseif problem == "MxV"
        A = rand(Float32, N, N)
        B = rand(Float32, N, 1)
        Nop = 2 * N^2
    elseif problem == "VxV"
        A = rand(Float32, N, 1)
        B = rand(Float32, N, 1)
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

function time_multiplication_ns(problem, N, N_cores)
    Time = zeros(length(N))
    for (i, n) in enumerate(N)
        A, B, Nop = init(problem, n)
        t1 = time_ns()
        mult(problem, A, B)
        t2 = time_ns()
        dt = t2 - t1
        Time[i] = dt / Nop

        println("N=", n, " Time per operation = ", Time[i], " ns")
    end
    return Time
end

function time_multiplication_belapsed(problem, N, N_cores)
    Time = zeros(length(N))
    for (i, n) in enumerate(N)
        A, B, Nop = init(problem, n)
        dt = @belapsed mult($problem, $A, $B)
        Time[i] = (dt / Nop) * 1e9  # Convert seconds to nanoseconds per operation

        println("N=", n, " Time per operation = ", Time[i], " ns")
    end
    return Time
end

function plot_GFLOPS_comparison()
    cpuid = cpuinfo()
    string_cpuid = string(cpuid)
    AVX_value = get_avx_value(string_cpuid)
    N_threads = Threads.nthreads()
    N_cores = N_threads / 2
    N = Vector([25:25:500; 600:50:1000])

    BLAS.set_num_threads(N_threads)

    Time_ns = time_multiplication_ns("MxM", N, N_cores)
    Time_belapsed = time_multiplication_belapsed("MxM", N, N_cores)

    GFLOPS_ns = 1 ./ Time_ns
    GFLOPS_belapsed = 1 ./ Time_belapsed

    plot(N, GFLOPS_ns, label = "Using time_ns", lw = 2)
    plot!(N, GFLOPS_belapsed, label = "Using @belapsed", lw = 2, linestyle = :dash)
    xlabel!("Matrix Dimension (N)")
    ylabel!("GFLOPS")
    title!("GFLOPS time_ns vs @belapsed")
end

plot_GFLOPS_comparison()

# Save the plot as SVG
savefig("plot_timens_vs_belapsed.png")
