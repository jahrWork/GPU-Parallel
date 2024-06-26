using CUDA
using Plots

# Prepare arrays to store the results
Ns = []    # tamaños
Ts = []    # tiemp

N_ops = 2 * 10000.0^3
for N = 300:25:2499 
    TIMES = div(N_ops, 2 * N^3)

    A = CUDA.rand(Float32, N, N)
    B = CUDA.rand(Float32, N, N)
    
    t = @elapsed begin
        for i = 1:TIMES
            A * B
        end
    end
    println("$N, $TIMES, $t")
    
    push!(Ns, N)
    push!(Ts, t)
end

# Plot the results
plot(Ns, Ts, label="Duration vs Matrix Size", xlabel="Size of the matrix N", ylabel="Duration (s)", title="Performance")
savefig("tiempogpu.png")