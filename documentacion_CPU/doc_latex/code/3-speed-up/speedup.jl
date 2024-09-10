import Pkg
Pkg.activate(".")
Pkg.add( "PGFPlotsX" ) 
using PGFPlotsX
using LinearAlgebra, MKL
using Plots

N = 10_000
A = rand(Float32, N, N)
B = rand(Float32, N, N)

function matrix_multiplication(A, B)
    return A * B
end

N_threads = [1, 2, 3, 4, 5, 6]
times = Float64[]

matrix_multiplication(A, B)

# Calculation of the reference for speed-up
BLAS.set_num_threads(1)
reference_time = @elapsed matrix_multiplication(A, B)

# Calculation of speed-up for different numbers of threads
speedups = Float64[]
for (i, threads) in enumerate(N_threads)
    BLAS.set_num_threads(threads)
        
    t = @elapsed matrix_multiplication(A, B)
    push!(times, t)
    speedup = reference_time / t
    push!(speedups, speedup)
    println("Threads: $threads, Time: $t, Speedup: $speedup")
end


x = N_threads
y = speedups

plot = @pgf Axis(
    {
        xlabel="Number of threads in use",
        ylabel="Speedup factor",
        title="Speedup",
        #legend="north east"
    },
    Plot({no_marks, "blue"}, Table(x, y)),
)

PGFPlotsX.save("/Users/juanromanbermejo/Desktop/documentacion_CPU/doc_latex/code/3-speed-up/speed-up.tex", plot, include_preamble=false)
