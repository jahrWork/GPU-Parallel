#using plots
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

tamaños_mat = [100, 300, 800]
tiempos = zeros(length(tamaños_mat))

for (i, N) in enumerate(tamaños_mat)
    A = rand(N, N)
    B = rand(N, N)
    tiempo = @elapsed multiplicar_matrices_amano(A, B)
    tiempos[i] = tiempo
    println("Ha tardado ", tiempo, " segundos para una matriz de tamaño ", N)
end



#plot(tamaño_mat,tiempo, xlabel"tamaño",ylabel"tiempo(s)")

