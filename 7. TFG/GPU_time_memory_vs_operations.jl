using LinearAlgebra, CUDA
using BenchmarkTools
using Printf

include("system_info.jl")  # Include the system_info.jl file for max CPU and GPU GFLOPS


# Operación de multiplicación de matrices en GPU
function matmul_gpu(Nt, A, B, C)
	for j in 1:Nt
		C = j * A * B
	end
	return C
end

function mul_gpu(Nt, A, B, C)

	for j in 1:Nt
		mul!(C, A, B)
	end

	return C

end


function matmul_kernel!(C, A, B, N, Nt)
	# Índice global lineal del thread
    idx = (blockIdx().x - 1)*blockDim().x + threadIdx().x #Ver GPU_indexing_kernel.jl para entender cómo se calcula
    stride = blockDim().x * gridDim().x # Calcula el salto para determinar cómo repartir las operaciones entre los hilos.
    
    # Se reparten las operaciones entre los hilos tal que no se solapen.
    for t in idx:stride:Nt
        @inbounds for i in 1:N
            for j in 1:N
                temp = 0.0f0
                for k in 1:N
					# Suma de los productos de los elementos de la fila i de A por los elementos de la columna j de B
                    temp += A[i, k] * B[k, j]
                end
                C[i, j] = temp
            end
        end
    end
    return
end

# Función para lanzar el kernel
function custom_gpu!(Nt, A::CuArray, B::CuArray, C::CuArray)
    N = size(A, 1)
    threads = 128  # Número de hilos por bloque (debe ser múltiplo de 32 y menor de 1024, el número máximo de threads por bloque)
    blocks = cld(Nt, threads)  # Número de bloques necesarios

    @cuda threads=threads blocks=blocks matmul_kernel!(C, A, B, N, Nt)
    return C
end


# Benchmarking sin uso de hilos de CPU
function measure_gpu(operations, Nt, N, Nop)
	# Crear matrices en GPU
	A = CUDA.randn(Float32, N, N)
	B = CUDA.randn(Float32, N, N)
	C = CUDA.zeros(Float32, N, N)

	# Warm-up
	x = operations(Nt, A, B, C)

	###### Forma 1 de medir tiempo con @belapsed ######
	# dt_belapsed = @belapsed $operations($Nt, $A, $B, $C)
	# Time_belapsed = (dt_belapsed / Nop) * 1e9  # Convertir segundos a nanosegundos
	# GFLOPS_belapsed = round(1 / Time_belapsed, digits = 2)

	CUDA.synchronize()

	###### Forma 2 de medir tiempo con CUDA.@elapsed ######
	dt_cu_elapsed = CUDA.@elapsed operations(Nt, A, B, C)
	Time_cu_elapsed = (dt_cu_elapsed / Nop) * 1e9  # Convertir segundos a nanosegundos
	GFLOPS_cu_elapsed = round(1 / Time_cu_elapsed, digits = 2)

	# Pretty print resultados
	# pretty_print(N, Nt, operations, GFLOPS_belapsed, "GPU")
	pretty_print(N, Nt, operations, GFLOPS_cu_elapsed, "GPU")

	#return GFLOPS_belapsed, GFLOPS_cu_elapsed
    return GFLOPS_cu_elapsed
end


# Función para imprimir resultados de forma ordenada
function pretty_print(c1, c2, c3, c4, c5)


	@printf("%10s %10s %20s %15s  %10s  \n", c1, c2, c3, c4, c5)


end

# Ejecución de pruebas
println("Pretty_Print GPU GFLOPS")

println(" ")


# Obtain maximum GPU GFLOPS
max_gpu_gflops = round(gpu_info(true), digits = 2)
println("Theoretical max GPU GFLOPS: ", round(max_gpu_gflops, digits = 2), " GFLOPS")

println(" ")

pretty_print("N", "Nt", "Operations", "GFLOPS", "Device")
dims = [(50, 100000), (100, 10000), (200, 1000), (400, 100), (800, 10)]
test = [matmul_gpu, mul_gpu, custom_gpu!]
for (N, Nt) in dims
	for f in test
		GFLOPS = measure_gpu(f, Nt, N, 2 * N^3 * Nt)
	end
	println(" ")
end


