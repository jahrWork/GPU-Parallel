n = 8000
file = open("datos.csv", "w")  # Abrir archivo para escritura

# Escribir encabezado en el archivo CSV
write(file, "Tamaño de la matriz, Duración (s)\n")

for i in 0:100:n
    a = rand(Float32, i, i)
    b = rand(Float32, i, i)
    inicio_ns = 0
    final_ns = 0
    duracion_s = 0
    inicio_ns = time_ns()
    c = a * b
    final_ns = time_ns()
    duracion_s = (final_ns - inicio_ns) / 1e9 # Convertir nanosegundos a segundos
    
    # Imprimir el número de n junto con el tiempo de duración
    println("n: $i, Duración: $duracion_s segundos")

    # Escribir datos en el archivo CSV
    write(file, "$i, $duracion_s\n")
end

close(file)  # Cerrar archivo
