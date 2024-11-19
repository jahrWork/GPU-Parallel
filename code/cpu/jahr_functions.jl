
import Pkg 
#Pkg.update()
# Pkg.add("BLAS")
#using LinearAlgebra, BLAS
# import Pkg
# Pkg.add("MKL")
# Pkg.instantiate()
#using LinearAlgebra, MKL
# Pkg.add(["CPUTime", "Plots", "LinearAlgebra", "MKL", "PGFPlotsX", "CpuId"])

# Pkg.add("LoopVectorization")
# Pkg.add("Octavian")
#using CPUTime, Plots, LinearAlgebra, MKL, PGFPlotsX, CpuId
using CPUTime, Plots, LinearAlgebra, MKL, CpuId
using Pkg
using LoopVectorization
using Octavian
using Tullio
using Strided
using Distributed
using Base.Threads
using BenchmarkTools


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

    (N, M) = size(A)
    (M, L) = size(B)
    C = zeros(Float32, (N, L))
    
    for k in 1:Nt
        C = k * Octavian.matmul(A, B)
    end

    return C
end

function matrix_mult__________strided(A, B)
    C = zeros(Float32, size(A, 1), size(B, 2))
    @strided C .= A * B
    return C
end


function matrix_mult______distributed(A, B)
    # Distributed.jl @distributed info:
    # The specified range is partitioned and locally executed across all workers.
    # In case an optional reducer function is specified, @distributed performs local reductions
    # on each worker with a final reduction on the calling process.

    # Note that without a reducer function, @distributed executes asynchronously,
    # i.e. it spawns independent tasks on all available workers and returns immediately
    # without waiting for completion. To wait for completion, prefix the call with @sync
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

function matrix______________id_check(A, B)
    (N, M) = size(A)
    (M, L) = size(B)
    C = zeros(Float32, (N, L))

    @threads for i in 1:nthreads()
        BLAS.set_num_threads(1)  # Asegurarse de que cada *thread* use un solo núcleo
        for j in i:nthreads():N  # Distribuir las filas entre los *threads*
            for k in 1:L
                C[j, k] = dot(A[j, :], B[:, k])
            end
        end
        println("Thread $(threadid()) processing part of the matrix.")
    end

    return C
end

function matrix_________custom_shared(A, B)
    (N, M) = size(A)
    (M, L) = size(B)
    BT = transpose(B)
    C = zeros(Float32, (N, L))
    
    # Dividir el trabajo en bloques de filas
    num_threads = nthreads()
    rows_per_thread = div(N, num_threads)  # Filas por hilo
    extra_rows = N % num_threads  # Resto de filas si no es divisible
    printed = false  # Variable de control para impresión única
    
    # Usar un @threads loop para cada hilo
    Threads.@threads for t in 1:num_threads
        # Asignar rango de filas que procesará este hilo
        start_row = (t - 1) * rows_per_thread + 1
        end_row = t * rows_per_thread
        if t == num_threads && extra_rows > 0  # Último hilo toma las filas extra
            end_row += extra_rows
        end

        println("Thread $(t) processing rows $(start_row):$(end_row).")

        t1 = time_ns()

        # Repetir el cálculo asignado Nt veces
        for repeat = 1:Nt            
            # Calcular el bloque de filas asignado
            for j in start_row:end_row
                for k in 1:L
                    C[j, k] = dot(A[j, :], B[:, k])
                end
            end
        end
    end
    
    return C
end

function mult______no_new_allocations(A, B, C)
    (N, M) = size(A)
    (M, L) = size(B)

    # Set BLAS to single-thread mode
    BLAS.set_num_threads(1)

    Threads.@threads for k in 1:Nt
        # Perform in-place multiplication to avoid allocations
        mul!(C, A, B)
    end

    return C
end

function mult_1_nt_no_new_allocations(A, B, C)
    (N, M) = size(A)
    (M, L) = size(B)
    
    Threads.@threads for k in 1:Nt
        BLAS.set_num_threads(1)
        C = k * A * B
    end
    return C
end

function matrix_mult2(A, B)
    (N, M) = size(A)
    (M, L) = size(B)
    C = zeros(Float32, (N, L))

    
    for k in 1:Nt
        C = k * A * B
    end

    return C
end

function mul__________________(A, B)
    (N, M) = size(A)
    (M, L) = size(B)
    C = zeros(Float32, (N, L))

    # Set BLAS to single-thread mode
    BLAS.set_num_threads(1)

    Threads.@threads for k in 1:Nt
        # Perform in-place multiplication to avoid allocations
        mul!(C, A, B)
    end

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




N = 50
A = rand(Float32, N, N)
B = rand(Float32, N, N)
C = zeros(Float32, size(A, 1), size(B, 2))
Nt = 10000


#N_threads = [1, 2, 4, 8, 16, 32]
#N_threads = [ 4 ]
matmul_functions = (
    # (my_matrix_multiplication, 2*N^3), 
    # (my_efficient_matrix_multiplication, 2*N^3),
    # (my_efficient_matrix_multiplication2, 2*N^3),
    (mult_No_parallel_1_thread___, 2 * N^3 * Nt),
    (mult_Nt_parallel_1_thread___, 2 * N^3 * Nt),
    (mult______no_new_allocations, 2 * N^3 * Nt),
    (mult_Nt_times_parallel______, 2 * N^3 * Nt),

    # (matrix_mult____________alloc, 2 * N^3),
     (matrix_mult_________________, 2 * N^3),
    # (matrix_mult____________turbo, 2 * N^3),
    # (matrix_mult___________tullio, 2 * N^3),
    # (matrix_mult_________octavian, 2 * N^3* Nt),
    # (matrix_mult__________strided, 2 * N^3),
    # (matrix_mult______distributed, 2 * N^3),
    # (matrix_mult_distributed_sync, 2 * N^3),
    #(matrix__________________mul!, 2 * N^3),
    # (matrix______________id_check, 2 * N^3),
    # (matrix_________custom_shared, 2 * N^3 * Nt),
    # (matrix_mult____________alloc, 2 * N^3),
    # (matrix_mult_________________, 2 * N^3),
    # (matrix_mult____________turbo, 2 * N^3),
    # (matrix_mult___________tullio, 2 * N^3),
    # (matrix_mult_________octavian, 2 * N^3),
    # (matrix_mult__________strided, 2 * N^3),
    # (matrix_mult______distributed, 2 * N^3),
    # (matrix_mult_distributed_sync, 2 * N^3),
    # (matrix__________________mul!, 2 * N^3),
    # (matrix______________id_check, 2 * N^3),
    # (matrix_________custom_shared, 2 * N^3 * Nt),
    (mult_1_nt_no_new_allocations, 2 * N^3 * Nt), 
    (mul__________________, 2 * N^3 * Nt ), 
    (matrix_mult2, 2 * N^3 * Nt ), 
  

)



cpuid = cpuinfo()
string_cpuid = string(cpuid)
println("AVX support: ", occursin("256", string_cpuid))
println("AVX-512 support: ", occursin("512 bit", string_cpuid))

for (mult, Nop) in matmul_functions
    #println("\nRunning: ", mult)

    # Benchmark the function with three arguments if it's `mult______no_new_allocations`
    if mult == mult______no_new_allocations || mult == mult_1_nt_no_new_allocations
        benchmark_result = @benchmark $mult($A, $B, $C)
    else
        benchmark_result = @benchmark $mult($A, $B)
    end

    N_threads = Threads.nthreads()
    N_cores = N_threads / 2

    AVX_value = get_avx_value(string_cpuid)
    Theoretical_time = 1e9 / (4.5e9 * AVX_value * 2 * N_cores)
    global GFLOPS_max = 1 / Theoretical_time

    # warm up
    if mult == mult______no_new_allocations || mult == mult_1_nt_no_new_allocations
        mult(A, B, C)
    else
        mult(A, B)
    end

    # Set the number of BLAS threads based on the number of cores
    BLAS.set_num_threads(N_threads)

    if mult == mult______no_new_allocations || mult == mult_1_nt_no_new_allocations
        t1 = time_ns()
        mult(A,B,C)
       # dt = @belapsed $mult($A, $B, $C)
        t2 = time_ns()
        dt = (t2 -t1)/1e9
    else
        #dt = @belapsed $mult($A, $B)
        t1 = time_ns()
        mult(A,B)
        t2 = time_ns()
        dt = (t2 -t1)/1e9
    end
    
    # Convert elapsed time to GFLOPS
    Time = dt  # Time is directly in seconds from @belapsed
    GFLOPS = Nop / (Time * 1e9)  # Convert to GFLOPS by dividing by 10^9
    println(mult, " N =", N, " Nt =", Nt, "    GFLOPS = ", round(GFLOPS; digits=0), "    num_threads = ", BLAS.get_num_threads())

    # # Obtain the median allocations and memory usage
    # median_allocations = median(benchmark_result).allocs
    # median_memory = round(median(benchmark_result).memory / 1024, digits=2)
    
    # # Print allocations and memory metrics
    # println("Allocations: ", median_allocations)
    # println("Memory allocated: ", median_memory, " KB")
end


println("\n GFLOPS_max = ", round(GFLOPS_max; digits=0))