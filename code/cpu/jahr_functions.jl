
# import Pkg 
# Pkg.add("BLAS")
#using LinearAlgebra, BLAS
import Pkg
# Pkg.add("MKL")
# Pkg.instantiate()
#using LinearAlgebra, MKL
#Pkg.add(["CPUTime", "Plots", "LinearAlgebra", "MKL", "PGFPlotsX", "CpuId"])

#Pkg.add("LoopVectorization")
#using CPUTime, Plots, LinearAlgebra, MKL, PGFPlotsX, CpuId
using CPUTime, Plots, LinearAlgebra, MKL, CpuId
using Pkg
using LoopVectorization
using Octavian
using Tullio
using Strided
using Distributed


function my_matrix_multiplication(A, B)

    (N, M) = size(A)
    (M, L) = size(B)

    C = zeros(Float32, (N, L))

    for i in 1:N, j in 1:L
        for k in 1:M
            C[i, j] = C[i, j] + A[i, k] * B[k, j]
        end
    end

    return C

end

function my_efficient_matrix_multiplication(A, B)

    (N, M) = size(A)
    (M, L) = size(B)
    BT = transpose(B)

    C = zeros(Float32, (N, L))

    for k in 1:M
        for j in 1:L, i in 1:N

            C[i, j] = C[i, j] + A[i, k] * BT[j, k]

        end
    end

    return C

end

function my_efficient_matrix_multiplication2(A, B)

    (N, M) = size(A)
    (M, L) = size(B)
    BT = transpose(B)

    C = zeros(Float32, (N, L))


    Threads.@threads for k in 1:M
        for j in 1:L, i in 1:N

            C[i, j] = C[i, j] + A[i, k] * BT[j, k]

        end
    end

    return C
end

function mult_Nt_parallel_1_thread___(A, B)
    (N, M) = size(A)
    (M, L) = size(B)
    C = zeros(Float32, (N, L))
    Threads.@threads for k in 1:Nt
        BLAS.set_num_threads(1)
        C = k * A * B
    end
    return C
end

function mult_No_parallel_1_thread___(A, B)
    (N, M) = size(A)
    (M, L) = size(B)
    C = zeros(Float32, (N, L))
    BLAS.set_num_threads(1)
    for k in 1:Nt
        C = k * A * B
    end
    return C
end

function mult_Nt_times_parallel______(A, B)
    (N, M) = size(A)
    (M, L) = size(B)
    C = zeros(Float32, (N, L))
    for k in 1:Nt
        C = k * A * B
    end
    return C
end

function matrix_mult_________________(A, B)
    return A * B
end

function matrix_mult____________alloc(A, B)
    (N, M) = size(A)
    (M, L) = size(B)
    C = zeros(Float32, (N, L))
    C = A * B
    return C
end

function matrix_mult____________turbo(A, B)
    @assert size(A, 2) == size(B, 1)
    m, n, p = size(A, 1), size(A, 2), size(B, 2)
    C = zeros(Float32, (m, p))
    @turbo for i = 1:m, j = 1:p, k = 1:n
        C[i, j] += A[i, k] * B[k, j]
    end
    return C
end

function matrix_mult___________tullio(A, B)
    C = zeros(Float32, size(A, 1), size(B, 2))
    @tullio C[i, j] = A[i, k] * B[k, j]
    return C
end

function matrix_mult_________octavian(A, B)
    C = Octavian.matmul(A, B)
    return C
end

function matrix_mult__________strided(A, B)
    C = zeros(Float32, size(A, 1), size(B, 2))
    @strided C .= A * B
    return C
end

function matrix_mult______distributed(A, B)
    C = zeros(Float32, size(A, 1), size(B, 2))
    @distributed for i in axes(A, 1)
        for j in axes(B, 2)
            C[i, j] = dot(A[i, :], B[:, j])
        end
    end
    return C
end

function matrix_mult_distributed_sync(A, B)
    C = zeros(Float32, size(A, 1), size(B, 2))
    @sync @distributed for i in axes(A, 1)
        for j in axes(B, 2)
            C[i, j] = dot(A[i, :], B[:, j])
        end
    end
    return C
end

function matrix__________________mul!(A, B)
    C = mul!(similar(A, Float32, size(A, 1), size(B, 2)), A, B)
    return C
end



function get_avx_value(string_cpuid)
    # Inicializar la variable AVX_Value
    AVX_value = 0

    # Buscar el size del vector SIMD en la cadena y asignar el valor correspondiente
    if occursin("256 bit", string_cpuid)
        AVX_value = 8
    elseif occursin("512 bit", string_cpuid)
        AVX_value = 16
    else
        AVX_value = 0
    end

    return AVX_value

end




N = 100
A = rand(Float32, N, N)
B = rand(Float32, N, N)
Nt = 10000


#N_threads = [1, 2, 4, 8, 16, 32]
#N_threads = [ 4 ]
matmul_functions = (
    # (my_matrix_multiplication, 2*N^3), 
    # (my_efficient_matrix_multiplication, 2*N^3),
    # (my_efficient_matrix_multiplication2, 2*N^3),
    (mult_No_parallel_1_thread___, 2 * N^3 * Nt),
    (mult_Nt_parallel_1_thread___, 2 * N^3 * Nt),
    (mult_Nt_times_parallel______, 2 * N^3 * Nt),
    (matrix_mult____________alloc, 2 * N^3),
    (matrix_mult_________________, 2 * N^3),
    (matrix_mult____________turbo, 2 * N^3),
    (matrix_mult___________tullio, 2 * N^3),
    # (matrix_mult_________octavian, 2 * N^3),
    # (matrix_mult__________strided, 2 * N^3),
    # (matrix_mult______distributed, 2 * N^3),
    # (matrix_mult_distributed_sync, 2 * N^3),
    (matrix__________________mul!, 2 * N^3)
)



cpuid = cpuinfo()
string_cpuid = string(cpuid)
println("AVX support: ", occursin("256", string_cpuid))
println("AVX-512 support: ", occursin("512 bit", string_cpuid))

for (mult, Nop) in matmul_functions

    mult(A, B)

    N_threads = Threads.nthreads()
    N_cores = N_threads / 2


    AVX_value = get_avx_value(string_cpuid)
    Theoretical_time = 1e9 / (4.5e9 * AVX_value * 2 * N_cores)
    global GFLOPS_max = 1 / Theoretical_time



    # Set the number of BLAS threads based on the number of cores
    BLAS.set_num_threads(N_threads)
    t1 = time_ns()
    mult(A, B)
    t2 = time_ns()
    dt = t2 - t1

    Time = dt / Nop
    GFLOPS = 1 / Time
    # println( "GFLOPS = ", GFLOPS, " N =", N, "  threads =", threads, " time =", dt  )
    println(mult, " N =", N, " Nt =", Nt, "    GFLOPS = ", round(GFLOPS; digits=0), "    num_threads = ", BLAS.get_num_threads())
    #end

end
println("\n GFLOPS_max = ", round(GFLOPS_max; digits=0))