using Base.Threads
using LinearAlgebra  # Importar la librería necesaria para BLAS
using Plots  # Para hacer el gráfico
using MKL

# Function to initialize matrices based on the problem type
function init(problem, N) 
    if problem == "MxM" 
        A = rand(Float32, N, N)
        B = rand(Float32, N, N)
        Nop = 2 * N^3  # Operaciones en multiplicación de matrices
    elseif problem == "MxV"
        A = rand(Float32, N, N)
        B = rand(Float32, N)
        Nop = 2 * N^2  # Operaciones en multiplicación matriz por vector
    elseif problem == "VxV"
        A = rand(Float32, N)
        B = rand(Float32, N)
        Nop = 2 * N  # Operaciones en multiplicación de vectores
    end 

    return A, B, Nop 
end

# Function to perform matrix multiplication
function mult(problem, A, B)
    if problem == "MxM" 
         return A * B  
    elseif problem == "MxV"
         return A * B
    elseif problem == "VxV"
         return transpose(A) * B
    end 
end

# Function to create the chunk distribution for each CPU
function distribute_chunks(max_N::Int, nthreads_::Int)
    cpu_chunks = [[] for _ in 1:nthreads_]  # Crear una lista de listas vacías para cada hilo

    for i in 10:max_N
        cpu_index = mod(i - 10, nthreads_) + 1  # Asignar a los hilos en orden circular
        push!(cpu_chunks[cpu_index], i)  # Agregar el tamaño de matriz correspondiente al hilo
    end

    return cpu_chunks
end

# Function to time matrix multiplication operations in parallel
function time_multiplication_parallel(problem, max_N)

    # Usamos max_N - 9 ya que el rango de N empieza en 10
    Time = zeros(max_N - 9)
    GFLOPS = zeros(max_N - 9)  # Para almacenar los GFLOPS
    
    nthreads_ = Threads.nthreads()  # Obtener número de threads
    cpu_chunks = distribute_chunks(max_N, nthreads_)  # Generar la distribución de chunks automáticamente
    
    Threads.@threads for tid in 1:nthreads_
        chunk = cpu_chunks[tid]  # Asignar un conjunto de dimensiones a cada hilo
        for dim in chunk
            A, B, Nop = init(problem, dim)

            t1 = time_ns()
            mult(problem, A, B)
            t2 = time_ns()
            
            dt = t2 - t1
            Time[dim - 9] = dt / 1e9  # Tiempo en segundos, ajustando índice (N empieza en 10)
            GFLOPS[dim - 9] = Nop / Time[dim - 9] / 1e9  # Calcular GFLOPS ajustando índice
            
            println("Thread $(threadid()): N=", dim, " GFLOPS =", GFLOPS[dim - 9])
        end
    end
    
    return GFLOPS, Time
end

# Function to time matrix multiplication operations in parallel
function time_multiplication_parallel(problem, max_N)

    # Usamos max_N - 9 ya que el rango de N empieza en 10
    Time = zeros(max_N - 9)
    GFLOPS = zeros(max_N - 9)  # Para almacenar los GFLOPS
    
    nthreads_ = Threads.nthreads()  # Obtener número de threads
    cpu_chunks = distribute_chunks(max_N, nthreads_)  # Generar la distribución de chunks automáticamente
    
    Threads.@threads for tid in 1:nthreads_
        chunk = cpu_chunks[tid]  # Asignar un conjunto de dimensiones a cada hilo
        for dim in chunk
            A, B, Nop = init(problem, dim)

            t1 = time_ns()
            mult(problem, A, B)
            t2 = time_ns()
            
            dt = t2 - t1
            Time[dim - 9] = dt / 1e9  # Tiempo en segundos, ajustando índice (N empieza en 10)
            GFLOPS[dim - 9] = Nop / Time[dim - 9] / 1e9  # Calcular GFLOPS ajustando índice
            
            println("Thread $(threadid()): N=", dim, " GFLOPS =", GFLOPS[dim - 9])
        end
    end
    
    return GFLOPS, Time
end

# Function to plot the GFLOPS
function plot_GFLOPS(N, GFLOPS)
    plot(N, GFLOPS, 
        title = "GFLOPS vs Matrix Size", 
        xlabel = "Matrix Size (N)", 
        ylabel = "GFLOPS",
        lw = 2, 
        label = "GFLOPS",
        legend = :topright)
end

# Call the function with parallelized outer loop
max_N = 200  # Puedes cambiarlo al valor máximo de N que desees
N = collect(10:max_N)

N_cores = Threads.nthreads()  # Utilizar todos los threads disponibles

# Ensure each multiplication happens in a single core
BLAS.set_num_threads(1)

# Calcular GFLOPS
GFLOPS, Time = time_multiplication_parallel("MxM", max_N)

# Graficar los GFLOPS
plot_GFLOPS(N, GFLOPS)
