n=8000
for i=0:100:n
    a=rand(Float64, i, i)
    b=rand(Float64, i, i)
    inicio_ns=0
    final_ns=0
    duracion_s=0
    inicio_ns = time_ns()
    c=a*b
    final_ns = time_ns()
    duracion_s = (final_ns - inicio_ns) / 1e9 # Convertir nanosegundos a segundos
    println("El tiempo de ejecución de la matriz rango $(i) fue: $(duracion_s) segundos")
end