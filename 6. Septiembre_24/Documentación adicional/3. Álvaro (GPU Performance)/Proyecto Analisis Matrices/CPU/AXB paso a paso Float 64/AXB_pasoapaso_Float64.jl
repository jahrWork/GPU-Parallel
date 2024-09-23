n = 1000
for i in 0:100:n
    inicio_ns = 0
    final_ns = 0
    duracion_s = 0
    a = rand(Float64, i, i)
    b = rand(Float64, i, i)
    c = zeros(Float64, i, i)
    inicio_ns = time_ns()
    if i > 0
        for j = 1:i
            for k = 1:i
                t = 0.0
                for x = 1:i
                    c[j,k] = t + a[j,x] * b[x,k]
                    t = c[j,k]
                end
            end
        end
    end
    final_ns = time_ns()
    duracion_s = (final_ns - inicio_ns) / 1e9 
    println("El tiempo de ejecución de la matriz de rango $(i) fue: $(duracion_s) segundos")
end
