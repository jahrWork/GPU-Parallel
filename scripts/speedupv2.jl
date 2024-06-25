using LinearAlgebra, Plots

function thread_speedup_benchmark(matrix_size::Int, max_threads::Int)
    A = rand(matrix_size, matrix_size)
    B = rand(matrix_size, matrix_size)
    times = zeros(max_threads)
    speedups = zeros(max_threads)

    println("Matrix Size: $matrix_size x $matrix_size")

   
    BLAS.set_num_threads(1)
    baseline_time = @elapsed A * B
    times[1] = baseline_time
    speedups[1] = 1  

    println("Threads: 1, Time: $baseline_time seconds")

  
    for threads in 2:max_threads
        BLAS.set_num_threads(threads)
        time_elapsed = @elapsed A * B
        times[threads] = time_elapsed
        speedups[threads] = baseline_time / time_elapsed
        println("Threads: $threads, Time: $time_elapsed seconds, Speedup: $(speedups[threads])")
    end

  
    p = plot(1:max_threads, speedups, title="Speedup by Number of Threads", xlabel="Number of Threads", ylabel="Speedup", legend=false, marker=:circle)
    display(p)
end


thread_speedup_benchmark(10000, 12)
