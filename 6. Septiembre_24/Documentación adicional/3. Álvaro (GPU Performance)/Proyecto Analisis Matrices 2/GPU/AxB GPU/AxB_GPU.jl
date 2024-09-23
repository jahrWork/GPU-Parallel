using CUDA
open("datos_MM_NCTE_RAW.csv", "w") do file
write(file, "Tamaño de la matriz, Duración (s)\n")

    for N = 100:100:10000

       A = CUDA.rand(Float32, N, N)
       B = CUDA.rand(Float32, N, N)
       t = CUDA.@elapsed A * B

       println("$(N), $(t)")
       #write(file, "$N, $t\n")
    end
end