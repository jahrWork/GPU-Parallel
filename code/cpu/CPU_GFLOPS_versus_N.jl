

using CPUTime, Plots, LinearAlgebra, MKL, BenchmarkTools, CpuId


function matmul_1t(Nt, A, B, C)
  
    BLAS.set_num_threads(1)
    Threads.@threads for k in 1:Nt 
        C =  k * A * B 
    end
  
    return C 
  end 

function init(N)

  A = rand(Float32, N, N)
  B = rand(Float32, N, N)
  Nop = 2 * N^3
  return A, B, Nop 

end

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



function plot_GFLOPS()

  # CPU Features
	cpuid = cpuinfo()
	string_cpuid = string(cpuid)
	println("AVX support: ", occursin("256", string_cpuid))
	println("AVX-512 support: ", occursin("512 bit", string_cpuid))

	AVX_value = get_avx_value(string_cpuid)

	# Number of cores
	println("num threads =", Threads.nthreads())
	N_threads = Threads.nthreads()
	N_cores = N_threads / 2
  # Time the matrix multiplication and matrix-vector multiplication operations
	Theoretical_time = 1e9 / (4.5e9 * AVX_value * 2 * N_cores)
  GFLOPS_max = 1 / Theoretical_time

 
  N =  Vector( [50:25:100;] )

  GFLOPS = zeros(size(N))


  for (i, n) in enumerate(N) 
       
        A, B, Nop = init(n)
        C = zeros(Float32, n, n)
        Nt = 100 
        Nop = Nop * Nt

		dt = @belapsed matmul_1t($Nt, $A, $B, $C)
		
		Time = (dt / Nop) * 1e9  # Convert seconds to nanoseconds
        GFLOPS[i] = 1 / Time 
        println( "N =", n, " GFLOPS = ", GFLOPS[i] )
  end  
     
  
  plot(N,  GFLOPS, 
       title = "GFLOPS matmul versus N", 
       xlabel = "\$ N \$", ylabel = "\$ GFLOPS \$", 
       ylimits=(0,maximum(GFLOPS)+100)
      )
  # Añadir Theoretical
	plot!(N, GFLOPS_max * ones(length(N)), label = "Theoretical", lw = 3)
  savefig("plot_GFLOPS_prueba.png")

end 

plot_GFLOPS()
