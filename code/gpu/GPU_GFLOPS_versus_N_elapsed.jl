using CUDA, Plots, LinearAlgebra

########################################################
####################### GPU INFO #######################
########################################################

function get_architecture_name(capability)
    if capability.major == 9
        return "Ada Lovelace"  # RTX 40 series
    elseif capability.major == 8
        return "Ampere"  # RTX 30 series
    elseif capability.major == 7
        if capability.minor == 5
            return "Turing"  # RTX 20 series
        else
            return "Volta"  # Tesla V100
        end
    elseif capability.major == 6
        return "Pascal"  # GTX 10 series
    elseif capability.major == 5
        return "Maxwell"  # GTX 900 series
    elseif capability.major == 3
        return "Kepler"  # GTX 600/700 series
    else
        return "Unknown Architecture"
    end
end

function cuda_cores_per_sm(capability)
    if capability.major == 9  # Ada Lovelace architecture (e.g., RTX 40 series)
        return 128
    elseif capability.major == 8  # Ampere architecture (e.g., RTX 30 series)
        return 128
    elseif capability.major == 7  # Turing/Volta architecture
        return 64
    elseif capability.major == 6  # Pascal architecture
        return 64
    else
        return 32  # Default to older architectures
    end
end

function gpu_info()
    device = CUDA.device()
    capability = CUDA.capability(device)
    sm_count = CUDA.attribute(device, CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    cores_per_sm = cuda_cores_per_sm(capability)  # Get cores per SM based on compute capability
    total_cuda_cores = sm_count * cores_per_sm

    println(" ")
    println("---------------------------------")
    println(" ")
    println("Device Information ")
    println(" ")
    println("GPU Name: ", CUDA.name(device))
    println("GPU Compute Capability: ", capability.major, ".", capability.minor, " (", get_architecture_name(capability), ")")
    println(" ")
    println("GPU Memory: ", CUDA.totalmem(device) / 1e9, " GB (base-10, where 1 GB = 1,000,000,000 bytes)")
    println("GPU Memory: ", CUDA.totalmem(device) / 2^30, " GiB (base-2, where 1 GiB = 1,073,741,824 bytes)")
    println(" ")
    println("GPU Streaming Multiprocessor (SM) Count: ", sm_count)
    println("CUDA Cores per SM: ", cores_per_sm)
    println("Total CUDA Cores: ", total_cuda_cores)
    println(" ")
    println("GPU Clock Rate: ", CUDA.attribute(device, CUDA.DEVICE_ATTRIBUTE_CLOCK_RATE) / 1e6, " GHz")
    println(" ")
    println("---------------------------------")
    println(" ")
end

function matmul_gpu(Nt, A, B, C)
    for k in 1:Nt
        C .= k .* (A * B)
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

function plot_GFLOPS_GPU()
    gpu_info() # Print GPU information

    N = collect(50:25:1000)
    GFLOPS = zeros(length(N))

    for (i, n) in enumerate(N)
        A, B, Nop = init(n)
        C = CUDA.zeros(Float32, n, n)
        Nt = 100
        Nop_total = Nop * Nt

        dt = CUDA.@elapsed begin
            matmul_gpu(Nt, A, B, C)
        end

        GFLOPS[i] = (Nop_total / dt) / 1e9
        println("N = $n, GFLOPS = $(GFLOPS[i])")
    end

    plot(N, GFLOPS,
        title = "GFLOPS de matmul en GPU vs N",
        xlabel = "\$ N \$", ylabel = "\$ GFLOPS \$",
        ylims = (0, maximum(GFLOPS) + 100),
    )
    savefig("plot_GFLOPS_GPU.png")
end

plot_GFLOPS_GPU()
