using BenchmarkTools, Statistics, LinearAlgebra

# Definir los valores iniciales
N = 50  # Tamaño de la matriz

println("Ejecutando benchmark para N = $N")

# Generar matrices A y B
A = rand(Float32, N, N)
B = rand(Float32, N, N)

# Calcular número de operaciones de punto flotante
Nop = 2 * N^3  # Número de operaciones de punto flotante

# Ajustar el tiempo máximo de benchmark
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 250
BenchmarkTools.DEFAULT_PARAMETERS.samples = 100
BenchmarkTools.DEFAULT_PARAMETERS.evals = 100000

# Realizar el benchmark
times = []
t = @benchmarkable $A * $B  # Usamos $ para hacer referencia a las matrices generadas
result = run(t)             # Ejecutar el benchmark
push!(times, median(result.times))  # Guardar el tiempo mínimo
println("Tiempo medio para esta iteración: $(median(result.times)) ns")


# Calcular el tiempo promedio
avg_time_ns = mean(times)
println("Tiempo promedio: $avg_time_ns ns")

# Calcular GFLOPS
time_s = avg_time_ns / 1e9  # Convertir nanosegundos a segundos
gflops = (Nop / time_s) / 1e9  # Convertir a GFLOPS
println("GFLOPS: $gflops")

# Imprimir el resultado final
println("Resultado final de GFLOPS para N=$N: $gflops")
