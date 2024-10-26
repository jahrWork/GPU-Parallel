using BenchmarkTools, Statistics, LinearAlgebra

# Definir los valores iniciales
N = 512  # Tamaño de la matriz
num_iterations = 3  # Número de iteraciones

println("Ejecutando benchmark para N = $N, con $num_iterations iteraciones...")

# Generar matrices A y B
A = rand(Float32, N, N)
B = rand(Float32, N, N)
println("Matrices A y B generadas.")

# Calcular número de operaciones de punto flotante
Nop = 2 * N^3  # Número de operaciones de punto flotante
println("Número de operaciones de punto flotante: $Nop")

# Ajustar el tiempo máximo de benchmark
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 250
println("Parámetro de benchmark ajustado a 250 segundos.")

# Realizar el benchmark
times = []
t = @benchmarkable $A * $B  # Usamos $ para hacer referencia a las matrices generadas
result = run(t)             # Ejecutar el benchmark
push!(times, median(result.times))  # Guardar el tiempo mínimo
println("Tiempo medio para esta iteración: $(median(result.times)) ns")

# Limpiar memoria
GC.gc()
println("Memoria limpiada tras benchmark.")

# Calcular el tiempo promedio
avg_time_ns = mean(times)
println("Tiempo promedio: $avg_time_ns ns")

# Calcular GFLOPS
time_s = avg_time_ns / 1e9  # Convertir nanosegundos a segundos
gflops = (Nop / time_s) / 1e9  # Convertir a GFLOPS
println("GFLOPS: $gflops")

# Imprimir el resultado final
println("Resultado final de GFLOPS para N=$N: $gflops")
