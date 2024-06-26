
using CUDA
using Plots

function matrix_initialization(N)
    A = CUDA.rand(Float32, N, N)
    B = CUDA.rand(Float32, N, N)
    return A, B
end 

function matrix_multiplication(A, B)
    C = CUDA.zeros(Float32, size(A,1), size(B,2))
    CUDA.@sync begin
       C = A * B
    end
    return C
end


N_sizes = [10:10:2500; 2500:100:10000]


times = Float64[]

for N in N_sizes
    A, B = matrix_initialization(N)
    start_time = CUDA.@elapsed matrix_multiplication(A, B)
    push!(times, start_time)
end

# Graficar los resultados
plot(N_sizes, times, title="GPU Matrix Multiplication Performance", xlabel="Matrix Size (N)", ylabel="Time (seconds)", legend=false, marker=:circle, yscale=:log10)

savefig("pruebas")