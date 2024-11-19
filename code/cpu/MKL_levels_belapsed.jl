#import Pkg
#Pkg.activate(".")
#Pkg.instantiate()
#Pkg.add(["CPUTime", "Plots", "LinearAlgebra", "MKL", "PGFPlotsX", "CpuId"])
#using CPUTime, Plots, LinearAlgebra, MKL, PGFPlotsX, CpuId
using CPUTime, Plots, LinearAlgebra, MKL, CpuId, BenchmarkTools

function get_avx_value(string_cpuid)
	# Inicializar la variable AVX_Value
	AVX_value = 0

	# Buscar el size del vector SIMD en la cadena y asignar el valor correspondiente
	if occursin("256 bit", string_cpuid)
		AVX_value = 8
	elseif occursin("512 bit", string_cpuid)
		AVX_value = 16
	else
		AVX_value = 0
	end

	return AVX_value
end


function init(problem, N)

	if problem == "MxM"
		A = rand(Float32, N, N)
		B = rand(Float32, N, N)
		Nop = 2 * N^3
	elseif problem == "MxV"
		A = rand(Float32, N, N)
		B = rand(Float32, N, 1)
		Nop = 2 * N^2
	elseif problem == "VxV"
		A = rand(Float32, N, 1)
		B = rand(Float32, N, 1)
		Nop = 2 * N
	end

	return A, B, Nop
end

function mult(problem, A, B)

	if problem == "MxM"
		return A * B
	elseif problem == "MxV"
		return A * B
	elseif problem == "VxV"
		return transpose(A) * B
	end

end


# Function to time matrix multiplication operations
function time_multiplication(problem, N, N_cores)
	Time = zeros(length(N))
	println("Problem = ", problem)

	for (i, n) in enumerate(N)
		A, B, Nop = init(problem, n)

		# Use @belapsed for time measurement, interpolate variables
		dt = @belapsed mult($problem, $A, $B)

		# Calculate time per operation in nanoseconds
		Time[i] = (dt / Nop) * 1e9  # Convert seconds to nanoseconds

		# Uncomment this if you want to print time per operation for each matrix size
		println("N=", n, " Time per operation = ", Time[i], " ns")
	end

	return Time
end



function plot_GFLOPS()

	# CPU Features
	cpuid = cpuinfo()
	string_cpuid = string(cpuid)
	println("AVX support: ", occursin("256", string_cpuid))
	println("AVX-512 support: ", occursin("512 bit", string_cpuid))

	AVX_value = get_avx_value(string_cpuid)
	println("AVX_value = ", AVX_value)

	# Number of cores
	println("num threads =", Threads.nthreads())
	N_threads = Threads.nthreads()
	N_cores = N_threads / 2

	# Range of matrix dimensions to test
	N = Vector([25:25:500; 600:50:1000; 1200:200:2000; 2500:500:5000])
	N = Vector([25:25:500; 600:50:1000])

	# Set the number of BLAS threads based on the number of cores
	BLAS.set_num_threads(N_threads)
	println("threads = ", BLAS.get_num_threads(), " N_cores =", N_cores)
	println(" ")

	# Time the matrix multiplication and matrix-vector multiplication operations
	Theoretical_time = 1e9 / (4.5e9 * AVX_value * 2 * N_cores)
	println("Theoretical time per operation = ", Theoretical_time, " ns")
	println("Max GFLOPS = ", 1 / Theoretical_time)
	println(" ")
	Time1 = time_multiplication("MxM", N, N_cores)
	Time2 = time_multiplication("MxV", N, N_cores)
	Time3 = time_multiplication("VxV", N, N_cores)

	# Calculate GFLOPS (floating-point operations per second)
	GFLOPS1 = 1 ./ Time1
	GFLOPS2 = 1 ./ Time2
	GFLOPS3 = 1 ./ Time3
	GFLOPS_max = 1 / Theoretical_time

	# Data for plotting
	x = float(N)

	GFLOPS4 = fill(GFLOPS_max, length(GFLOPS1))
	max1 = maximum(GFLOPS1)
	max2 = maximum(GFLOPS2)
	max3 = maximum(GFLOPS3)
	println(" ")
	println("Theoretical_Max_GFLOPS = ", GFLOPS_max)
	println("max1 = Max_GFLOPS Mat x Mat = ", max1)
	println("max2 = Max_GFLOPS Mat x Vect = ", max2)
	println("max3 = Max_GFLOPS Vect x Vect = ", max3)
	println(" ")
	println("Ratio max1 / max2 = ", max1 / max2)
	println("Ratio max1 / max3 = ", max1 / max3)
	println(" ")

	# Primer plot
	plot(N, GFLOPS1,
		title = "GFLOPS versus N - @belapsed",
		xlabel = "\$ N \$", ylabel = "GFLOPS",
		label = "Mat x Mat", lw = 3,
		xlimits = (0, 5000), ylimits = (0, max1 + 100),
	)

	# Añadir Mat x Vect
	plot!(N, GFLOPS2, label = "Mat x Vect", lw = 3)

	# Añadir Vect x Vect
	plot!(N, GFLOPS3, label = "Vect x Vect", lw = 3)

	# Añadir Theoretical
	plot!(N, GFLOPS4, label = "Theoretical", lw = 3)


  # Save the plot as SVG
  savefig("plot_GFLOPS_belapsed.png")

  # Segundo plot con zoom
	plot(N, GFLOPS1,
  title = "GFLOPS versus N - Zoom @belapsed",
  xlabel = "\$ N \$", ylabel = "GFLOPS",
  label = "Mat x Mat", lw = 3,
  xlimits = (0, 1000), ylimits = (0, max1 + 100),
  )

  # Añadir Mat x Vect
  plot!(N, GFLOPS2, label = "Mat x Vect", lw = 3)

  # Añadir Vect x Vect
  plot!(N, GFLOPS3, label = "Vect x Vect", lw = 3)

  # Añadir Theoretical
  plot!(N, GFLOPS4, label = "Theoretical", lw = 3)

	# Save the plot as SVG
	savefig("plot_GFLOPS_zoom_belapsed.png")

	# println(x)
	# println(y1)

	# plot(x, y1)
	# plot!(x, y2)
	# plot!(x, y3)
	# plot!(x, y4)

	# plot = @pgf Axis(
	#     {
	#         width = "15cm",  
	#         height = "10cm", 
	#         xlabel="Matrix dimension",
	#         ylabel="GFLOPS",
	#         title="[M]x[M] vs [M]x[v]",
	#         legend="north east",
	#         ymax=500,

	#     },
	#     Plot({no_marks, "blue"}, Table(x, y1)),
	#     Plot({no_marks, "red"}, Table(x, y2)),
	#     Plot({no_marks, "green"}, Table(x, y3)),
	#     Plot({no_marks, "orange"}, Table(x, y4)),
	#     LegendEntry("Matmul"),
	#     LegendEntry("MatVec"),
	#     LegendEntry("VecVec"),
	#     LegendEntry("Theoretical"),
	# )

	# display(plot)
	#PGFPlotsX.save("code/BLAS_levels_dyn.tex", plot, include_preamble=false)


end





plot_GFLOPS()



