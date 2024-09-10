import Pkg
Pkg.activate(".")
Pkg.add( "PGFPlotsX" )   # environment in this folder
using CPUTime
using Plots
using LinearAlgebra, MKL
using PGFPlotsX


function matrix_initialization(N)

    A = rand(Float32, N, N )
    B = rand(Float32, N, N )
    return A, B 

end 


function matrix_multiplication(A,B)

   return A * B  
  
end


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


function time_matrix_multilication(N, N_cores, matmul)

  Time = zeros( length(N) )
  #Theoretical_time = 4e9/(4e9 * 512/32 * 4 * N_cores)
  Theoretical_time = 1e9/(4e9 * 512/32 * 2 * N_cores) # jahr

  for (i,n) in enumerate(N)  # variables inside loop have local scope 
 
   A,B = matrix_initialization(n)

   t1= time_ns()

   matmul(A,B)
   
   t2 = time_ns()
   Time[i] = (t2-t1)/(2*n^3)
   

   println("N=", n, " Time per operation =", Time[i] , " nsec")
   println("N=", n, " Theoretical time per operation =", Theoretical_time, " nsec")
    
  end 
  
  return Time, Theoretical_time

end 

# settings "julia.NumThreads": "auto"
N_threads = Threads.nthreads()
N_cores =  div(N_threads, 2)
println("Threads =", N_threads ) 
println("Cores =", N_cores )
time_matrix_multilication(2000, N_cores, matrix_multiplication)


N = 0:100:2500
BLAS.set_num_threads(2*N_cores)
println(" threads = ", BLAS.get_num_threads(), " N_cores =", N_cores )
Time, Theoretical_time = time_matrix_multilication(N, N_cores, matrix_multiplication)
Time2, Theoretical_time2 = time_matrix_multilication(N, N_cores, my_matrix_multiplication)
GFLOPS = 1 ./ Time
GFLOPS2 = 1 ./ Time2

x = N
y1 = GFLOPS
y2 = GFLOPS2

grafico = @pgf Axis(
    {
        xlabel="Dimensión de la matriz [N]",
        ylabel="Operaciones por segundo [GFLOPS]",
        title="Comparison of different dot functions",
        #legend="north east"
 # Posición de la leyenda
    },
    Plot({no_marks, "blue"}, Table(x, y1)),  # Primera serie (Seno)
    Plot({no_marks, "red"}, Table(x, y2)),   # Segunda serie (Coseno)
)

# Guardar el gráfico en un archivo .tex
PGFPlotsX.save("/Users/juanromanbermejo/Desktop/codigo/grafico.tex", grafico)

#display( plot(N, GFLOPS, ylims=(0, 5000), title="GFLOPS",  minorgrid=true  ) )
#display( plot!(N, GFLOPS_max *ones( length(N) ), minorgrid=true ) )
