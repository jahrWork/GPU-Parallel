using Plots

function multiplicar_matrices_fila_a_fila(A, B)

    
    m, n = size(A, 1), size(B, 2)
    p = size(A, 2)
    
    C = zeros(eltype(A), m, n)
    
    for i in 1:m
        for j in 1:n
            for k in 1:p
                C[i, j] += A[i, k] * B[k, j]
            end
        end
    end
    
    return C
end

function multiplicar_matrices_columna_a_columna(A, B)
 
    
    m, n = size(A, 1), size(B, 2)
    p = size(A, 2)
    
    C = zeros(eltype(A), m, n)
    
    for j in 1:n
        for i in 1:m
            for k in 1:p
                C[i, j] += A[i, k] * B[k, j]
            end
        end
    end
    
    return C
end

function medir_tiempo_multiplicacion(N)
    A = rand(N, N)
    B = rand(N, N)
    
    tiempo_fila_a_fila = @elapsed multiplicar_matrices_fila_a_fila(A, B)
    tiempo_columna_a_columna = @elapsed multiplicar_matrices_columna_a_columna(A, B)
    
    println("Para N = ", N, ":")
    println("Tiempo de multiplicación fila a fila: ", tiempo_fila_a_fila, " segundos")
    println("Tiempo de multiplicación columna a columna: ", tiempo_columna_a_columna, " segundos")
    
    return tiempo_fila_a_fila, tiempo_columna_a_columna
end

# Crear una matriz para almacenar los resultados
resultados = zeros(2, 5)

# Realizar mediciones y almacenar resultados
for i in 1:3
    N = 100 * i
    fila_a_fila, columna_a_columna = medir_tiempo_multiplicacion(N)
    resultados[:, i] = [fila_a_fila, columna_a_columna]
end

# Graficar los resultados
tamaños_matriz = [100, 200, 300]
plot(tamaños_matriz, resultados', label=["Fila a Fila" "Columna a Columna"], xlabel="Tamaño de la Matriz (N)", ylabel="Tiempo (segundos)", title="Comparación de Tiempos de Multiplicación")

# Imprimir los resultados
println("\nResultados:")
println(resultados)
