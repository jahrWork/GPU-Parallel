#import Pkg
#Pkg.activate(".")
#Pkg.add(["CPUTime", "Plots", "LinearAlgebra", "MKL", "PGFPlotsX", "CpuId"])
#using CPUTime, Plots, LinearAlgebra, MKL, PGFPlotsX, CpuId
using CPUTime, Plots, LinearAlgebra, MKL, CpuId

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


function init(problem, N) 

    if problem == "MxM" 
        A = rand(Float32, N, N )
        B = rand(Float32, N, N )
        Nop = 2 * N^3 
    elseif problem == "MxV"
        A = rand(Float32, N, N )
        B = rand(Float32, N, 1 )
        Nop = 2 * N^2 
    elseif problem == "VxV"
        A = rand(Float32, N, 1 )
        B = rand(Float32, N, 1 )
        Nop = 2 * N 
    end 

    return A, B, Nop 
end 

function mult(problem, A,B)

    if problem == "MxM" 
         return  A * B  
    elseif problem == "MxV"
         return A * B
    elseif problem == "VxV"
         return  transpose(A) * B
    end 

end


# Function to time matrix multiplication operations
function time_multiplication(problem, N, N_cores)

    Time = zeros( length(N) )
    
    for (i,n) in enumerate(N)  
   
     A,B, Nop = init(problem, n)
     println("problem =", problem, " size A = ", size(A), "size B = ", size(B))
  
     t1 = time_ns()
     mult(problem, A, B)
     t2 = time_ns()
     dt = t2-t1
     
     Time[i] = dt / Nop
     
     println("N=", n, " Time per operation =", Time[i] , " nsec")
      
    end 
    
    return Time
  end


  function plot_GFLOPS() 

    # CPU Features
      cpuid = cpuinfo() 
      string_cpuid = string(cpuid)
      println("AVX support: ", occursin("256", string_cpuid))
      println("AVX-512 support: ", occursin("512 bit", string_cpuid))
  
      AVX_value = get_avx_value(string_cpuid)
  
    # Number of cores
      N_cores = 4
  
    # Range of matrix dimensions to test
      N = Vector([10:25:2500; 2500:100:5000])
  
    # Set the number of BLAS threads based on the number of cores
      BLAS.set_num_threads(2*N_cores) 
      println(" threads = ", BLAS.get_num_threads(), " N_cores =", N_cores )
  
    # Time the matrix multiplication and matrix-vector multiplication operations
      Theoretical_time = 1e9 /(4.5e9 * AVX_value * 2 * N_cores)
      println(" Theoretical time per operation =", Theoretical_time, " nsec")
      Time1 = time_multiplication("MxM", N, N_cores)
      Time2 = time_multiplication("MxV", N, N_cores)
      Time3 = time_multiplication("VxV", N, N_cores)
  
    # Calculate GFLOPS (floating-point operations per second)
      GFLOPS1 = 1 ./ Time1
      GFLOPS2 = 1 ./ Time2
      GFLOPS3 = 1 ./ Time3
      GFLOPS_max = 1 / Theoretical_time
  
    # Data for plotting
      x = float(N)
      
      GFLOPS4 = fill(GFLOPS_max, length(GFLOPS1))
      max1 =  maximum(GFLOPS1)   
      max2 =  maximum(GFLOPS2)     
      println("max GFLOPS Mat x Mat = ", max1 )  
      println("max GFLOPS Mat x Vect = ", max2 )
      println( "Ratio max1/ max2 = ", max1/max2 )
  
      plot!(N, [ GFLOPS1 GFLOPS2 GFLOPS3 GFLOPS4 ],  
              title = "GFLOPS versus number of operations", 
              xlabel = "\$ N \$", ylabel = "GFLOPS", 
              label = ["Mat x Mat" "Mat x Vect" "Vect x Vect" "Theoretical"], lw = 3,
              xlimits=(0,1000), ylimits=(0,300)
          )
  
      
  
      # println(x)
      # println(y1)
  
      # plot(x, y1)
      # plot!(x, y2)
      # plot!(x, y3)
      # plot!(x, y4)
  
      # plot = @pgf Axis(
      #     {
      #         width = "15cm",  
      #         height = "10cm", 
      #         xlabel="Matrix dimension",
      #         ylabel="GFLOPS",
      #         title="[M]x[M] vs [M]x[v]",
      #         legend="north east",
      #         ymax=500,
              
      #     },
      #     Plot({no_marks, "blue"}, Table(x, y1)),
      #     Plot({no_marks, "red"}, Table(x, y2)),
      #     Plot({no_marks, "green"}, Table(x, y3)),
      #     Plot({no_marks, "orange"}, Table(x, y4)),
      #     LegendEntry("Matmul"),
      #     LegendEntry("MatVec"),
      #     LegendEntry("VecVec"),
      #     LegendEntry("Theoretical"),
      # )
  
      # display(plot)
      #PGFPlotsX.save("code/BLAS_levels_dyn.tex", plot, include_preamble=false)
  
  
  end 






















  
function matrix_initialization(N) 


    A = rand(Float32, N, N )
    B = rand(Float32, N, N )
  
    return A, B 
  end 
  
  function matrix_vector_initialization(N)
  
      A = rand(Float32, N, N )
      B = rand(Float32, N, 1 )
    
      return A, B 
  end 
  
  function vector_vector_initialization(N)
  
      A = rand(Float32, N, 1 )
      B = rand(Float32, N, 1 )
    
      return A, B 
  end 
  
  
  
  
  
  
  
  function vector_multiplication(A,B)
  
      return dot(A, B)  
  end
  
  function matrix_multiplication(A,B)
  
      return  A * B  
  end
  
  function vector_multiplication(A,B)
  
      return  transpose(A) * B  
  end


# Function to time matrix multiplication operations
function time_matrix_multiplication(N, N_cores, matinit, matmul, AVX_value)

    Time = zeros( length(N) )
    #Se considera que solo se necesita 1 instruccion para FMA
    Theoretical_time = 1e9 /(4.5e9 * AVX_value * 2 * N_cores)
    #Theoretical_time = 2e9/(1.7e9 * 512/32 * 2 * N_cores)
  
    for (i,n) in enumerate(N)  
   
     A,B = matinit(n)
  
     t1 = time_ns()
     matmul(A,B)
     t2 = time_ns()
     dt = t2-t1
     
     Time[i] = dt/(2*n^3)
     
     println("N=", n, " Time per operation =", Time[i] , " nsec")
     println("N=", n, " Theoretical time per operation =", Theoretical_time, " nsec")
      
    end 
    
    return Time, Theoretical_time
  end



# Function to time matrix-vector multiplication operations
function time_matrix_vector_multiplication(N, N_cores, matinit, matmul)

    Time2 = zeros( length(N) )
  
    for (i,n) in enumerate(N) 
   
     A,B = matinit(n)
    
     t1 = time_ns()
     matmul(A,B)
     t2 = time_ns()
     dt = t2-t1
     
     Time2[i] = dt/(2*n^2)
  
     println("N=", n, " Time per operation =", Time2[i] , " nsec")
#     println("N=", n, " Theoretical time per operation =", Theoretical_time, " nsec")
      
    end 
    
    return Time2
end 

# Function to time matrix-vector multiplication operations
function time_vector_vector_multiplication(N, N_cores, matinit, matmul)

    Time3 = zeros( length(N) )
  
    for (i,n) in enumerate(N) 
   
     A,B = matinit(n)
    
     t1 = time_ns()
     matmul(A,B)
     t2 = time_ns()
     dt = t2-t1
     
     Time3[i] = dt/(2*n)
  
     println("N=", n, " Time per operation =", Time3[i] , " nsec")
   #  println("N=", n, " Theoretical time per operation =", Theoretical_time, " nsec")
      
    end 
    
    return Time3
end 


function plot_GFLOPS2() 

  # CPU Features
    cpuid = cpuinfo() 
    string_cpuid = string(cpuid)
    println("AVX support: ", occursin("256", string_cpuid))
    println("AVX-512 support: ", occursin("512 bit", string_cpuid))

    AVX_value = get_avx_value(string_cpuid)

  # Number of cores
    N_cores = 4

  # Range of matrix dimensions to test
    N = Vector([10:25:2500; 2500:100:5000])

  # Set the number of BLAS threads based on the number of cores
    BLAS.set_num_threads(2*N_cores) 
    println(" threads = ", BLAS.get_num_threads(), " N_cores =", N_cores )

  # Time the matrix multiplication and matrix-vector multiplication operations
    Time, Theoretical_time = time_matrix_multiplication(N, N_cores, matrix_initialization, matrix_multiplication, AVX_value)
    Time2 = time_matrix_vector_multiplication(N, N_cores, matrix_vector_initialization, matrix_multiplication)
    Time3 = time_vector_vector_multiplication(N, N_cores, vector_vector_initialization, vector_multiplication)

  # Calculate GFLOPS (floating-point operations per second)
    GFLOPS = 1 ./ Time
    GFLOPS2 = 1 ./ Time2
    GFLOPS3 = 1 ./ Time3
    GFLOPS_max = 1 / Theoretical_time

  # Data for plotting
    x = float(N)
    y1 = GFLOPS
    y2 = GFLOPS2
    y3 = GFLOPS3
    y4 = fill(GFLOPS_max, length(y1))
    max1 =  maximum(y1)   
    max2 =  maximum(y2)     
    println("max GFLOPS Mat x Mat = ", max1 )  
    println("max GFLOPS Mat x Vect = ", max2 )
    println( "Ratio max1/ max2 = ", max1/max2 )

    plot!(x, [ y1 y2 y3 y4 ],  
            title = "GFLOPS versus number of operations", 
            xlabel = "\$ N \$", ylabel = "GFLOPS", 
            label = ["Mat x Mat" "Mat x Vect" "Vect x Vect" "Theoretical"], lw = 3,
            xlimits=(0,1000), ylimits=(0,300)
        )

    

    # println(x)
    # println(y1)

    # plot(x, y1)
    # plot!(x, y2)
    # plot!(x, y3)
    # plot!(x, y4)

    # plot = @pgf Axis(
    #     {
    #         width = "15cm",  
    #         height = "10cm", 
    #         xlabel="Matrix dimension",
    #         ylabel="GFLOPS",
    #         title="[M]x[M] vs [M]x[v]",
    #         legend="north east",
    #         ymax=500,
            
    #     },
    #     Plot({no_marks, "blue"}, Table(x, y1)),
    #     Plot({no_marks, "red"}, Table(x, y2)),
    #     Plot({no_marks, "green"}, Table(x, y3)),
    #     Plot({no_marks, "orange"}, Table(x, y4)),
    #     LegendEntry("Matmul"),
    #     LegendEntry("MatVec"),
    #     LegendEntry("VecVec"),
    #     LegendEntry("Theoretical"),
    # )

    # display(plot)
    #PGFPlotsX.save("code/BLAS_levels_dyn.tex", plot, include_preamble=false)


end 


plot_GFLOPS2()

plot_GFLOPS()



