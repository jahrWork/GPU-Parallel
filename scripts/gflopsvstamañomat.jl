using Plots
using LinearAlgebra
using BenchmarkTools

# Función para calcular GFLOPS
function gflops(N, tiempo)
    operaciones = 2 * N^3
    gflops = operaciones / tiempo / 1e9
    return gflops
end

function main()
    # Rango de tamaños de matriz para probar
    Ns = 100:1000:10000
    gflops_results = []

    # Inicializar matrices A y B fuera del bucle
    global A = rand(100, 100)
    global B = rand(100, 100)

    for N in Ns
        global A, B
        A = rand(N, N)
        B = rand(N, N)
        bench = @benchmark A * B
        tiempo_medio = median(bench).time  # tiempo en nanosegundos
        push!(gflops_results, gflops(N, tiempo_medio))
    end

    # Crear el gráfico
    plot(Ns, gflops_results, title="GFLOPS vs Tamaño de Matriz N", xlabel="N", ylabel="GFLOPS", legend=false)
    savefig("gflops_vs_N.png")  # Guarda la gráfica en un archivo
end

main()  #
5