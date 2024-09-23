using Base.Threads

open("datos_MM_NCTE_RAW.csv", "w") do file  # Abrir archivo para escritura

    # Escribir encabezado en el archivo CSV
    write(file, "Tamaño de la matriz, Número de operaciones, Duración (s)\n")

    #Precision doble
    N_ops= 2 * 10000.0^3
    
    for N = 50:25:2499
        TIMES = div(N_ops, 2 * N^3)

        A = rand(Float32, N, N)
        B = rand(Float32, N, N)
        C = zeros(Float32, N, N)

        t0 = time_ns()

        n_threads = Threads.nthreads()  # Obtener el número de hilos disponibles

        Threads.@threads for i = 1:TIMES
            C .= A * B
        end

        t1 = time_ns()
        execution_time = (t1 - t0) / 1e9  # Convertir nanosegundos a segundos
        
        println("$N, $TIMES, $execution_time")

        # Escribir datos en el archivo CSV
        write(file, "$N, $TIMES, $execution_time\n")
    end
end
