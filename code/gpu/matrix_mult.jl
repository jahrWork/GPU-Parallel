import Pkg
Pkg.activate(".")
Pkg.add( "PGFPlotsX" )

using CUDA
using PGFPlotsX

# Initialize arrays to store N and GFLOPS values
x = Int[]         # Array to store N values
y1 = Float64[]    # Array to store GFLOPS values

for N = 100:100:4000 #limite superior 10000

    A = CUDA.rand(Float32, N, N)
    B = CUDA.rand(Float32, N, N)
    t = CUDA.@elapsed A * B

    # Compute GFLOPS
    gflops = (2 * N^3) / (t * 1e9)

    println("N = $(N), Time = $(t), GFLOPS = $(gflops)")

    # Store the values in arrays
    push!(x, N)
    push!(y1, gflops)

end

# Create the plot using PGFPlotsX
plot = @pgf Axis(
    {
        xlabel="Matrix dimension [N]",
        ylabel="GFLOPS",
        title="Matrix Multiplication Performance",
    },
    Plot({no_marks, "blue"}, Table(x, y1)),
)

PGFPlotsX.save("/ALVARO/Documentacion GPU/doc_latex/code/1/grafico_matrix_mult.tex", plot, include_preamble=false)
