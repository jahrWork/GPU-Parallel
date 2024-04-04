using CPUTime
using Plots
using LinearAlgebra, MKL

function matrix_initialization(N)
    A = rand(Float32, N, N)
    B = rand(Float32, N, N)
    return A, B
end

function benchmark_matrix_multiplication(filename, N, N_cores)
    Time = zeros(length(N))
    Theoretical_time = 4e9 / (4e9 * 512/32 * 4 * N_cores)

    open(filename, "w") do file
        write(file, "Tamaño de la matriz, Número de operaciones, Duración (s)\n")

        for (i, n) in enumerate(N)
            A, B = matrix_initialization(n)

            t1 = time_ns()
            C = A * B  # Usando la multiplicación de matrices de MKL
            t2 = time_ns()

            execution_time = (t2 - t1) / (2 * n^3)

            println("N=", n, " Time per operation =", execution_time, " nsec")
            println("N=", n, " Theoretical time per operation =", Theoretical_time, " nsec")

            write(file, "$n, $(2 * n^3), $execution_time\n")
            Time[i] = execution_time
        end
    end

    return Time, Theoretical_time
end

benchmark_matrix_multiplication("datos_MM_NCTE_RAW.csv", 50:50:1450, 1)

data = DelimitedFiles.readdlm("datos_MM_NCTE_RAW.csv", ',', skipstart=1)

matrix_size = data[:, 1]
num_operations = data[:, 2]
execution_time = data[:, 3]

scatter(matrix_size, execution_time, xlabel="Tamaño de la matriz", ylabel="Duración (s)", label="", title="Comparación de tiempo vs tamaño de la matriz")
plot!(matrix_size, execution_time, label="")  # Unir los puntos en el gráfico
