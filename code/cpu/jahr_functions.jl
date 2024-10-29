using LinearAlgebra
using Statistics
using BenchmarkTools
using Plots
using Printf

# Define functions
function mult_Nt_parallel_1_thread(A,B,Nt)
    (N, M) = size(A)
    (M, L) = size(B)
    C = zeros(Float32,  (N, L) )
   
    Threads.@threads for k in 1:Nt 
        # Removed BLAS.set_num_threads(1) from inside the loop
        C .=  k .* A * B 
    end
  
    return C 
end 

function mult_No_parallel_1_thread(A,B,Nt)
    (N, M) = size(A)
    (M, L) = size(B) 
    C = zeros(Float32,  (N, L) )
    BLAS.set_num_threads(1)
    for k in 1:Nt 
        C .=  k .* A * B 
    end
    return C 
end 

function mult_Nt_times_parallel___(A, B, Nt)
    (N, M) = size(A)
    (M, L) = size(B) 
    C = zeros(Float32,  (N, L) )
    for k in 1:Nt 
        C .=  k .* A * B 
    end
    return C 
end

function matrix_mult______________(A, B)
    return A * B
end

function  matrix_mult_________alloc(A, B)
    (N, M) = size(A)
    (M, L) = size(B) 
    C = zeros(Float32,  (N, L) )
    C .=  A * B 
    return C 
end

function main()
    # First Plot: N is constant, Nt increases
    N = 100
    A = rand(Float32, N, N)
    B = rand(Float32, N, N)
    Nt_list = [1, 10, 25, 50, 75, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000, 7500, 10000, 15000, 20000, 50000, 100000]
    
    matmul_functions = [
        (mult_No_parallel_1_thread, "No Parallel 1 Thread"), 
        (mult_Nt_parallel_1_thread, "Nt Parallel 1 Thread"),
        (mult_Nt_times_parallel___, "Nt Times Parallel")
    ]
    
    results = Dict()
    
    for (func, func_name) in matmul_functions
        results[func_name] = Dict("Nt" => Float64[], "GFLOPS" => Float64[])
        for Nt in Nt_list
            # Warm-up
            func(A, B, Nt)
            # Time measurement
            t = @elapsed func(A, B, Nt)
            # Calculate GFLOPS
            Nop = 2 * N^3 * Nt
            GFLOPS = Nop / (t * 1e9)
            push!(results[func_name]["Nt"], Nt)
            push!(results[func_name]["GFLOPS"], GFLOPS)
            @printf("Function: %-25s Nt: %-8d Time: %.4f s, GFLOPS: %.2f\n", func_name, Nt, t, GFLOPS)
        end
    end
    
    # Plotting the first graph
    plt1 = plot(title="Performance with N constant (N=$N)", xscale=:log10, yscale=:linear, dpi=600)
    for (func_name, data) in results
        plot!(plt1, data["Nt"], data["GFLOPS"], label=func_name, marker=:o)
    end
    xlabel!(plt1, "Nt")
    ylabel!(plt1, "GFLOPS")
    savefig(plt1, "plot1_N_constant.png")
    display(plt1)
    
    # Second Plot: Nt is constant, N increases
    Nt = 1000
    N_list = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
    
    matmul_functions = [
        (mult_No_parallel_1_thread, "No Parallel 1 Thread"), 
        (mult_Nt_parallel_1_thread, "Nt Parallel 1 Thread"),
        (mult_Nt_times_parallel___, "Nt Times Parallel"),
        (matrix_mult______________, "Matrix Mult"),
        (matrix_mult_________alloc, "Matrix Mult Alloc")
    ]
    
    results = Dict()
    
    for (func, func_name) in matmul_functions
        results[func_name] = Dict("N" => Float64[], "GFLOPS" => Float64[])
        for N in N_list
            local A = rand(Float32, N, N)
            local B = rand(Float32, N, N)
            if func === matrix_mult______________ || func === matrix_mult_________alloc
                # For functions without Nt parameter, run them Nt times
                # Warm-up
                for _ in 1:10
                    func(A, B)
                end
                # Time measurement
                t = @elapsed begin
                    for _ in 1:Nt
                        func(A, B)
                    end
                end
                Nop = 2 * N^3 * Nt
            else
                # Warm-up
                func(A, B, Nt)
                # Time measurement
                t = @elapsed func(A, B, Nt)
                Nop = 2 * N^3 * Nt
            end
            # Calculate GFLOPS
            GFLOPS = Nop / (t * 1e9)
            push!(results[func_name]["N"], N)
            push!(results[func_name]["GFLOPS"], GFLOPS)
            @printf("Function: %-25s N: %-6d Time: %.4f s, GFLOPS: %.2f\n", func_name, N, t, GFLOPS)
        end
    end
    
    # Plotting the second graph
    plt2 = plot(title="Performance with Nt constant (Nt=$Nt)", xscale=:linear, yscale=:linear, dpi=600)
    for (func_name, data) in results
        plot!(plt2, data["N"], data["GFLOPS"], label=func_name, marker=:o)
    end
    xlabel!(plt2, "N")
    ylabel!(plt2, "GFLOPS")
    savefig(plt2, "plot2_Nt_constant.png")  # Save second plot
    display(plt2)
end

main()
