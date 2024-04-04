using Plots
using DelimitedFiles



function benchmark_matrix_multiplication(filename)
    open(filename, "w") do file
        write(file, "Tamaño de la matriz, Número de operaciones, Duración (s)\n")

        N_ops = 2 * 10000.0^3

        for N = 50:25:2499
            TIMES = div(N_ops, 2 * N^3)
            

            t0 = time_ns()

            for i = 1:TIMES
                A = rand(Float32, N, N)
                B = rand(Float32, N, N)
                C = A * B
            end

            t1 = time_ns()
            execution_time = (t1 - t0) / 1e9 / TIMES 
            println("$N, $N_ops, $execution_time")
            write(file, "$N, $N_ops, $execution_time\n")
        end
    end
end

benchmark_matrix_multiplication("datos_MM_NCTE_RAW.csv")

data = DelimitedFiles.readdlm("datos_MM_NCTE_RAW.csv", ',', skipstart=1)

matrix_size = data[:, 1]
num_operations = data[:, 2]
execution_time = data[:, 3]

scatter(matrix_size, execution_time, xlabel="Tamaño de la matriz", ylabel="Duración (s)", label="", title="Comparación de tiempo vs tamaño de la matriz")