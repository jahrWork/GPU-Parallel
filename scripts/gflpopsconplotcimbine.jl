function plot_combined(N_threads_range)
    combined_plot = plot(title="Combined GFLOPS", xlabel="N", ylabel="GFLOPS", ylims=(0, 800), minorgrid=true)
   
    for N_threads in N_threads_range
        N_threads = N_threads
        N_cores = N_threads
        println("Threads =", N_threads )
        println("Cores =", N_cores )
       
        N = Vector([10:10:2500; 2500:100:5000])
        BLAS.set_num_threads(N_cores)
        # BLAS.set_num_threads(2*N_cores) # Duda sobre esto
        println(" threads = ", BLAS.get_num_threads(), " N_cores =", N_cores )
        Time, Theoretical_time = time_matrix_multilication(N, N_cores, matrix_multiplication)
        GFLOPS = 1 ./ Time
        GFLOPS_max = 1 / Theoretical_time
   
        plot!(combined_plot, N, GFLOPS, label="Threads = $N_threads", legend=:topleft)
        plot!(combined_plot, N, GFLOPS_max * ones(length(N)), label=false, linecolor=:black)
    end
    display(combined_plot)
  end
   
  plot_combined(1:8)