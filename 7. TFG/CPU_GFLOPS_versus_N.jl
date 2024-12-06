using Plots, LinearAlgebra, MKL, BenchmarkTools
include("system_info.jl")  # Include the system_info.jl file for max CPU and GPU GFLOPS

function matmul_1t(Nt, A, B, C)
	BLAS.set_num_threads(1)
	Threads.@threads for k in 1:Nt
		C = k * A * B
	end
	return C
end

function init(N)
	A = rand(Float32, N, N)
	B = rand(Float32, N, N)
	Nop = 2 * N^3
	return A, B, Nop
end

function plot_GFLOPS()
	N = Vector(50:25:100)
	GFLOPS = zeros(size(N))

	# Obtain maximum CPU GFLOPS
	max_cpu_gflops = round(cpu_info(true), digits = 2)
	println("Theoretical max CPU GFLOPS: ", round(max_cpu_gflops, digits = 2), " GFLOPS")
	println(" ")
	println("Measured GFLOPS:")
	println(" ")

	for (i, n) in enumerate(N)
		A, B, Nop = init(n)
		C = zeros(Float32, n, n)
		Nt = 100
		Nop = Nop * Nt

		dt = @belapsed matmul_1t($Nt, $A, $B, $C)

		Time = (dt / Nop) * 1e9  # Convert seconds to nanoseconds
		GFLOPS[i] = 1 / Time
		println("N = ", n, " GFLOPS = ", GFLOPS[i])
	end

	# Create the plot
	plot(
		N, GFLOPS,
		title = "GFLOPS matmul versus N",
		xlabel = "\$ N \$",
		ylabel = "\$ GFLOPS \$",
		ylimits = (0, max_cpu_gflops + 100),
		label = "Measured GFLOPS",
		lw = 3,
	)

	# Add Theoretical GFLOPS
	plot!(
		N, fill(max_cpu_gflops, length(N)),
		label = "Theoretical max GFLOPS = $max_cpu_gflops",
		lw = 3,
	)

	println(" ")

	savefig("7. TFG/plots/plot_GFLOPS_CPU.png")
end

plot_GFLOPS()
