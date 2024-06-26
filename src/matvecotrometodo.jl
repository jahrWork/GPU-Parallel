using Plots
using BenchmarkTools


function gflops(A, B, tiempo)
    return 2*size(A,1)*size(A,2)*length(B) / tiempo / 1e9
end


# Asumiendo un rendimiento teórico de pico de ~450 GFLOPS para operaciones de doble precisión
gflops_teoricos = 450


tamaños = [128, 256, 512, 1024, 2048, 4096]

# Almacenar los resultados
resultados = []

for N in tamaños
    A = rand(N, N)
    B = rand(N)
    C = zeros(N)

    tiempo = @belapsed for i in 1:$N
        $C = $A * $B
    end
    push!(resultados, gflops(A, B, tiempo))
end

# Crear la gráfica
plot(tamaños, resultados, label="GFLOPS medidos", xlabel="Tamaño de la matriz", ylabel="GFLOPS", lw=2)
hline!([gflops_teoricos], label="GFLOPS teóricos", lw=2)
