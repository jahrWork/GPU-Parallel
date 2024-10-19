using LinearAlgebra, BenchmarkTools, Plots, Base.Threads

#No funciona

function custom_matmul(A::Array{Float32,2}, B::Array{Float32,2})
    N = size(A, 1)
    C = Array{Float32}(undef, N, N)
    
    @inbounds @threads for i = 1:N
        for j = 1:N
            sum = 0.0f0
            for k = 1:N
                sum += A[i, k] * B[k, j]
            end
            C[i, j] = sum
        end
    end
    return C
end

function time_custom_matmul(N_range)
    GFLOPS = zeros(length(N_range))
    for (i, N) in enumerate(N_range)
        A = rand(Float32, N, N)
        B = rand(Float32, N, N)
        Nop = 2 * N^3

        # Definir el número de repeticiones
        num_reps = 10

        # Medir el tiempo usando BenchmarkTools
        t = @belapsed for _ in 1:$num_reps
            custom_matmul($A, $B)
        end

        # Calcular GFLOPS
        GFLOPS[i] = ((Nop * num_reps) / t) / 1e9

        println("N = $N, Tiempo Total = $(t*1e9) ns, GFLOPS = $(GFLOPS[i])")
    end
    return GFLOPS
end

function plot_GFLOPS_custom()
    N_range = collect(100:10:200)

    # Medir GFLOPS con implementación personalizada
    GFLOPS_custom = time_custom_matmul(N_range)

    # Calcular GFLOPS teóricos
    CPU_frequency_GHz = 3.9
    CPU_frequency = CPU_frequency_GHz * 1e9
    N_logical_cores = 12  # Usa todos los hilos de tu CPU
    AVX_value = 8

    Theoretical_GFLOPS = (CPU_frequency * AVX_value * 2 * N_logical_cores) / 1e9
    println("GFLOPS teóricos = $Theoretical_GFLOPS")

    # Graficar los resultados
    plot(N_range, GFLOPS_custom;
         title = "GFLOPS vs N (Implementación Personalizada)",
         xlabel = "N",
         ylabel = "GFLOPS",
         label = "GFLOPS Custom",
         lw = 3,
         legend=:bottomright)
    hline!([Theoretical_GFLOPS], label = "GFLOPS Teóricos", linestyle=:dash)
    display(current())
end

plot_GFLOPS_custom()
