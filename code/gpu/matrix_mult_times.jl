import Pkg
Pkg.activate(".")
Pkg.add( "PGFPlotsX" )

using CUDA
using PGFPlotsX

# Initialize arrays to store N and GFLOPS values
x = Int[]         # Array to store N values
y1 = Float32[]    # Array to store GFLOPS values

N_ops = 2 * 10000.0^3
for N = 2000:100:2499 #limite inf era 300, 25 el paso y 2499 limite superior 

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

# Create the plot using PGFPlotsX
plot = @pgf Axis(
    {
        xlabel="Matrix dimension [N]",
        ylabel="time [t]",
        title="Matrix Multiplication Performance Using Times",
    },
    Plot({no_marks, "blue"}, Table(x, y1)),
)

PGFPlotsX.save("/ALVARO/Documentacion GPU/doc_latex/code/2/grafico_matrix_mult_times.tex", plot, include_preamble=false)
