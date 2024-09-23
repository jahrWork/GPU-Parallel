open("datos_MM_NCTE_RAW.csv", "w") do file
    write(file, "Tamaño de la matriz, Número de operaciones, Duración (s)\n")

    N_ops = 2 * 10000.0^3

    for N = 50:25:2499
        TIMES = div(N_ops, 2 * N^3)

        A = rand(Float32, N, N)
        B = rand(Float32, N, N)
        C = zeros(Float32, N, N)

        t0 = time_ns()

        for i = 1:TIMES
            C .= i *  A * B
        end

        t1 = time_ns()
        execution_time = (t1 - t0) / 1e9  

        println("$N, $TIMES, $execution_time")
        write(file, "$N, $TIMES, $execution_time\n")
    end
end