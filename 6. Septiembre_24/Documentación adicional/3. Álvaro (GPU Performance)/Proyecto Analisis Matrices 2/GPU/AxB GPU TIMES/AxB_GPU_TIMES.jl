using CUDA

open("datos_MM_NCTE_RAW.csv", "w") do file 
write(file, "Tamaño de la matriz, Número de operaciones, Duración (s)\n")
    
    N_ops = 2 * 10000.0^3
    for N = 300:25:2499 
        TIMES = div(N_ops, 2 * N^3)

        A = CUDA.rand(Float32, N, N)
        B = CUDA.rand(Float32, N, N)
        
        t = @elapsed begin
            for i = 1:TIMES
                A * B
            end
        end
        println("$N, $TIMES, $t")
        write(file, "$N, $TIMES, $t\n")
    end

end