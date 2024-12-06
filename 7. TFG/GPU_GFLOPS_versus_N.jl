using Plots, LinearAlgebra, CUDA, BenchmarkTools
include("system_info.jl")  # Include the system_info.jl file for max CPU and GPU GFLOPS

function matmul_1t(Nt, A, B, C)
    for k in 1:Nt
        C = k * (A * B)
    end
    CUDA.synchronize()
    return C
end

function init(N)
    A = CUDA.rand(Float32, N, N)
    B = CUDA.rand(Float32, N, N)
    Nop = 2 * N^3
    return A, B, Nop
end

function plot_GFLOPS()
    N = Vector(50:25:1000)
    GFLOPS_belapsed = zeros(size(N))  # For @belapsed
    GFLOPS_cu_elapsed = zeros(size(N))  # For CUDA.@elapsed

    # Obtain maximum GPU GFLOPS
    max_gpu_gflops = round(gpu_info(true), digits = 2)
    println("Theoretical max GPU GFLOPS: ", round(max_gpu_gflops, digits = 2), " GFLOPS")
    println(" ")
    println("Measured GFLOPS:")
    println(" ")

    for (i, n) in enumerate(N)
        A, B, Nop = init(n)
        C = CUDA.zeros(Float32, n, n)
        Nt = 100
        Nop = Nop * Nt

        ###### Forma 1 de medir tiempo con @belapsed ######
        dt_belapsed = @belapsed matmul_1t($Nt, $A, $B, $C)
        Time_belapsed = (dt_belapsed / Nop) * 1e9  # Convert seconds to nanoseconds
        GFLOPS_belapsed[i] = 1 / Time_belapsed
        println("N = ", n, " GFLOPS (belapsed) = ", GFLOPS_belapsed[i])

        ###### Forma 2 de medir tiempo con CUDA.@elapsed ######
        dt_cu_elapsed = CUDA.@elapsed matmul_1t(Nt, A, B, C)
        Time_cu_elapsed = (dt_cu_elapsed / Nop) * 1e9  # Convert seconds to nanoseconds
        GFLOPS_cu_elapsed[i] = 1 / Time_cu_elapsed
        println("N = ", n, " GFLOPS (CUDA.@elapsed) = ", GFLOPS_cu_elapsed[i])
    end

    # Create the plot
    plot(
        N, GFLOPS_belapsed,
        title = "GFLOPS matmul versus N",
        xlabel = "\$ N \$",
        ylabel = "\$ GFLOPS \$",
        ylimits = (0, maximum(GFLOPS_belapsed) + 100),
        label = "Measured GFLOPS (@belapsed)",
        lw = 3,
    )

    # Add CUDA.@elapsed GFLOPS to the same plot
    plot!(
        N, GFLOPS_cu_elapsed,
        label = "Measured GFLOPS (CUDA.@elapsed)",
        lw = 3,
    )

    # Add Theoretical GFLOPS
    plot!(
        N, fill(max_gpu_gflops, length(N)),
        label = "Theoretical max GFLOPS: $max_gpu_gflops",
        lw = 3,
    )

    println(" ")

    savefig("7. TFG/plots/plot_GFLOPS_GPU.png")
end

plot_GFLOPS()
