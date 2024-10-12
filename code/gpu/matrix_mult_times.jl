using CUDA
using Plots

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
    println("GPU Memory: ", CUDA.totalmem(device) / 1e9, " GB")
    println("GPU Memory: ", CUDA.totalmem(device) / 2^30, " GiB") # Conversion to GiB
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

# Display GPU information
gpu_info()


########################################################
################# MATRIX MULTIPLICATION ################
########################################################

# Initialize arrays to store N and GFLOPS values
x = Int[]         # Array to store N values
y1 = Float32[]    # Array to store GFLOPS values

N_ops = 2 * 10000.0^3
for N = 2000:100:2499  # Limite inferior es 2000, paso de 100, y limite superior es 2499

    TIMES = div(N_ops, 2 * N^3)

    A = CUDA.rand(Float32, N, N)
    B = CUDA.rand(Float32, N, N)

    t = @elapsed begin
        for i = 1:TIMES
            A * B
        end
    end
    println("$N, $TIMES, $t")

    # Store the values in arrays
    push!(x, N)
    push!(y1, t)
end

# Create the plot using Plots.jl
plot(x, y1, label="Multiplication time", lw=2, title="Matrix Multiplication Performance Using Times",
    xlabel="Matrix dimension [N]", ylabel="Time [s]", legend=:topright)
