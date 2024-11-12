using CPUTime, LinearAlgebra, MKL, BenchmarkTools, Base.Threads

N = 50
Nt = 1000
A = rand(Float32, N, N)
B = rand(Float32, N, N)
C = zeros(Float32, size(A, 1), size(B, 2))

t_CPU = (2 * N^3 / (432 / 6 * 10^9)) * 10^6
t_memory = (3 * N^2 / (3200 * 10^6)) * 10^6

println("=====================================")
println(" ")
println("MxM Benchmark Nt parallel manual thread distribution") 
println("t_CPU: ", t_CPU, " μs")
println("t_memory: ", t_memory, " μs")
println(" ")

# Set BLAS to single-thread mode
BLAS.set_num_threads(1)

function matrix_mult_manual_threads(A, B, Nt)
    n_threads = Threads.nthreads()
    Nt_per_thread = div(Nt, n_threads)  # Number of iterations per thread
    remainder = Nt % n_threads  # Remainder to be added to the last thread if needed

    @sync begin
        for t in 1:n_threads
            Threads.@spawn begin
                local_Nt = if t == n_threads
                    Nt_per_thread + remainder
                else
                    Nt_per_thread
                end
                for i in 1:local_Nt
                    C = A * B
                end
            end
        end
    end
        
end

# Benchmark the parallel matrix multiplication
dt1 = @belapsed C = A * B  # Single-threaded baseline time
dt2 = ( @belapsed matrix_mult_manual_threads(A, B, Nt) ) / Nt

println("Time 1: ", dt1 * 10^6, " μs - GFLOPS: ", 2 * N^3 / (dt1 * 10^9))
println("Time 2: ", dt2 * 10^6, " μs - GFLOPS: ", 2 * N^3 / (dt2 * 10^9))
