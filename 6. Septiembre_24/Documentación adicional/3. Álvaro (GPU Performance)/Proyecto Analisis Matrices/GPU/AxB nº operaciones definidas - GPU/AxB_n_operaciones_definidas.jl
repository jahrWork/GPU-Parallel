using CUDA
device()

 A = CUDA.rand(3024,3024);
 B = CUDA.rand(3024,3024);
CUDA.@time A * B

 A = rand(3024,3024);
 B =rand(3024,3024);
@time A * B


using CUDA

N_ops = 2 * 10000.0^3

N = 1000
TIMES = div(N_ops, 2 * N^3)

A = CUDA.rand(Float32, N, N)
B = CUDA.rand(Float32, N, N)

tiempos = Float64[]

for i = 1:TIMES
    t = @elapsed begin
        CUDA.@sync C = A * B
    end
    push!(tiempos, t)
end

println("Tiempos de cada operación: ", tiempos)

tiempo_promedio = mean(tiempos)
println("Tiempo promedio de las operaciones: $tiempo_promedio s")
