using BenchmarkTools
using Plots

function multiplicar_matrices_amano(M1, M2)
    N = size(M1, 1)
    C = zeros(N, N)
    for i in 1:N
        for j in 1:N
            for m in 1:N
                C[i, j] += M1[i, m] * M2[m, j]
            end
        end
    end
    return C
end

tamaños_mat = [100, 200, 300]  # Añadimos más tamaños
tiempos = zeros(length(tamaños_mat))

for (i, N) in enumerate(tamaños_mat)
    A = rand(N, N)
    B = rand(N, N)
    tiempo = @belapsed multiplicar_matrices_amano($A, $B)  # Utilizamos @belapsed para obtener mediciones más precisas
    tiempos[i] = tiempo
    println("Ha tardado ", tiempo, " segundos para una matriz de tamaño ", N)
end

gr()
plot(tamaños_mat, tiempos, xlabel="Tamaño de la matriz (N)", ylabel="Tiempo (segundos)", label="Multiplicación de matrices")

display(Plots.current())