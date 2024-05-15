using BenchmarkTools
using CUDA
using Plots

function sum_scalar_cpu(n)
    s = 0.0
    for i = 1:n
        s += rand(Float64)
    end
    return s
end

function kernel(s, n)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    if i <= n
        @CUDA.atomic s[i] += rand(Float64)
    end
    return
end

function sum_scalar_gpu(n)
    s = CUDA.zeros(n)
    @cuda threads=256 kernel(s, n)
    return sum(s)
end

function benchmark_sum_scalar(n)
    times_cpu = Float64[]
    times_gpu = Float64[]
    gflops_cpu = Float64[]
    gflops_gpu = Float64[]
    
    for i in 1:n
        # Benchmark CPU
        time_cpu = @belapsed sum_scalar_cpu($i)
        push!(times_cpu, time_cpu)
        gflops_cpu_current = (2 * i) / (time_cpu * 1e9)
        push!(gflops_cpu, gflops_cpu_current)
        
        # Benchmark GPU
        time_gpu = @belapsed sum_scalar_gpu($i)
        push!(times_gpu, time_gpu)
        gflops_gpu_current = (2 * i) / (time_gpu * 1e9)
        push!(gflops_gpu, gflops_gpu_current)
    end
    
    return times_cpu, times_gpu, gflops_cpu, gflops_gpu
end



function plot_gflops(n)
    times_cpu, times_gpu, gflops_cpu, gflops_gpu = benchmark_sum_scalar(n)
    
    x = 1:n
    plot(x, gflops_cpu, label="CPU", xlabel="Scalars summed", ylabel="GFLOPS", lw=2)
    plot!(x, gflops_gpu, label="GPU", lw=2)
    title!("GFLOPS Comparison: CPU vs GPU")
    display(Plot)
end

plot_gflops(1000)  
