
# import Pkg 
# Pkg.add("BLAS")
#using LinearAlgebra, BLAS
#import Pkg 
# Pkg.add("MKL")
# Pkg.instantiate()
#using LinearAlgebra, MKL
#Pkg.add(["CPUTime", "Plots", "LinearAlgebra", "MKL", "PGFPlotsX", "CpuId"])
#using CPUTime, Plots, LinearAlgebra, MKL, PGFPlotsX, CpuId
using CPUTime, Plots, LinearAlgebra, MKL, CpuId

function my_matrix_multiplication(A,B)

    (N, M) = size(A)
    (M, L) = size(B) 
  
    C = zeros(Float32,  (N, L) )
  
    for i in 1:N, j in 1:L
        for k in 1:M
          C[i,j] = C[i,j] + A[i,k]*B[k,j]
        end  
    end
  
    return C 
  
end
  
function my_efficient_matrix_multiplication(A,B)
  
    (N, M) = size(A)
    (M, L) = size(B) 
    BT = transpose(B) 
  
    C = zeros(Float32,  (N, L) )
  
    for k in 1:M
      for j in 1:L, i in 1:N
        
          C[i,j] = C[i,j] + A[i,k]*BT[j,k]
  
      end  
    end
  
    return C 
  
end
  
function my_efficient_matrix_multiplication2(A,B)
  
    (N, M) = size(A)
    (M, L) = size(B) 
    BT = transpose(B) 
  
    C = zeros(Float32,  (N, L) )
    
  
    Threads.@threads for k in 1:M
      for j in 1:L, i in 1:N
        
          C[i,j] = C[i,j] + A[i,k]*BT[j,k]
          
      end  
    end
  
    return C 
end 

function mult_Nt_parallel_1_thread(A,B)
  
    (N, M) = size(A)
    (M, L) = size(B) 
    C = zeros(Float32,  (N, L) )
   

    Threads.@threads for k in 1:Nt 
        BLAS.set_num_threads(1)
        C =  k * A * B 
    end
  
    return C 
end 

function mult_No_parallel_1_thread(A,B)
  
  (N, M) = size(A)
  (M, L) = size(B) 
  C = zeros(Float32,  (N, L) )
 

      BLAS.set_num_threads(1)
      for k in 1:Nt 
        C =  k * A * B 
      end

  return C 
end 

function  mult_Nt_times_parallel___(A, B)

  (N, M) = size(A)
  (M, L) = size(B) 
  C = zeros(Float32,  (N, L) )
 

  for k in 1:Nt 
      C =  k * A * B 
  end

  return C 
end


function matrix_mult______________(A, B)


    return A * B

end

function  matrix_mult_________alloc(A, B)

  (N, M) = size(A)
  (M, L) = size(B) 
  C = zeros(Float32,  (N, L) )
 
      C =   A * B 

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





N = 700
A = rand(Float32, N, N)
B = rand(Float32, N, N)
Nt = 1000


#N_threads = [1, 2, 4, 8, 16, 32]
#N_threads = [ 4 ]
matmul_functions = ( 
    # (my_matrix_multiplication, 2*N^3), 
    # (my_efficient_matrix_multiplication, 2*N^3),
    # (my_efficient_matrix_multiplication2, 2*N^3),
    (mult_No_parallel_1_thread, 2*N^3*Nt), 

    (mult_Nt_parallel_1_thread, 2*N^3*Nt),
    (mult_Nt_times_parallel___, 2*N^3*Nt),

    (matrix_mult_________alloc, 2*N^3),
    (matrix_mult______________, 2*N^3)  )


cpuid = cpuinfo() 
string_cpuid = string(cpuid)
println("AVX support: ", occursin("256", string_cpuid))
println("AVX-512 support: ", occursin("512 bit", string_cpuid))

for (mult, Nop) in matmul_functions

     mult(A, B)

     N_threads = Threads.nthreads()
     N_cores = N_threads/2
     
  
     AVX_value = get_avx_value(string_cpuid)
     Theoretical_time = 1e9 /(4.5e9 * AVX_value * 2 * N_cores)
     global GFLOPS_max = 1 / Theoretical_time


  
  # Set the number of BLAS threads based on the number of cores
    BLAS.set_num_threads(N_threads) 
    t1 = time_ns()
    mult(A, B)
    t2 = time_ns()
    dt = t2-t1
    
    Time = dt / Nop 
    GFLOPS = 1 / Time 
   # println( "GFLOPS = ", GFLOPS, " N =", N, "  threads =", threads, " time =", dt  )
    println(  mult, " N =", N, " Nt =",  Nt,"    GFLOPS = ", round( GFLOPS; digits=0), "    num_threads = ", BLAS.get_num_threads() ) 
  #end

end 
println( "\n GFLOPS_max = ", round( GFLOPS_max; digits=0) )
