using MKL
using LinearAlgebra
using Random
using Base.Threads

# Dimensiones para las matrices
n = 100
iterations = 1000000

# Generar matrices aleatorias
A = rand(n, n)
B = rand(n, n)

println("A*B 100x100, 1M iteraciones")

# Imprimir información sobre el proveedor de BLAS
println("Proveedor BLAS: ", BLAS.vendor(), " (using ", BLAS.get_config(), ")")

# Función para multiplicar matrices con opciones de hilos
function test_matrix_multiplication(threads::Int, use_threads::Bool)
    C = zeros(n, n)  # Matriz resultado

    println(" ")

    # Establecer el número de hilos
    BLAS.set_num_threads(threads)
    println("Iniciando nueva configuración - Hilos BLAS establecidos en ", BLAS.get_num_threads())

    # Medir el tiempo de ejecución
    elapsed_time = @elapsed begin
        if use_threads
            Threads.@threads  for i in 1:iterations
                C .= A * B  # Asignación elemento a elemento para rendimiento
            end
        else
            for i in 1:iterations
                C .= A * B  # Asignación elemento a elemento para rendimiento
            end
        end
    end

    # Calcular el número de operaciones de punto flotante
    num_flops = 2 * n^3 * iterations

    # Calcular GFLOPS
    gflops = num_flops / elapsed_time / 1e9  # Convertir a GFLOPS

    return elapsed_time, gflops
end

# Probar diferentes combinaciones
configurations = [
    (1, false),  # 1 hilo, sin @threads
    (1, true),   # 1 hilo, con @threads
    (6, false),  # 6 hilos, sin @threads
    (6, true),   # 6 hilos, con @threads
    # (12, false), # 12 hilos, sin @threads
    # (12, true),  # 12 hilos, con @threads
]

for (threads, use_threads) in configurations
    elapsed_time, gflops = test_matrix_multiplication(threads, use_threads)
    println("Hilos: $threads, Uso de @threads: $use_threads")
    println("Tiempo transcurrido: $elapsed_time segundos")
    println("GFLOPS: $gflops")
end
