import Pkg
Pkg.activate(".")
Pkg.add( "PGFPlotsX" )  
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


function time_matrix_multilication(N, N_cores, matmul)

  Time = zeros( length(N) )
  #Theoretical_time = 4e9/(4e9 * 512/32 * 4 * N_cores)
  Theoretical_time = 2e9/(1.7e9 * 512/32 * 2 * N_cores)

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

N = Vector([10:10:2500; 2500:250:10000])
BLAS.set_num_threads(2*N_cores)
println(" threads = ", BLAS.get_num_threads(), " N_cores =", N_cores )
Time, Theoretical_time = time_matrix_multilication(N, N_cores, matrix_multiplication)
GFLOPS = 1 ./ Time
GFLOPS_max = 1 / Theoretical_time

x = N
y1 = GFLOPS
y2 = GFLOPS_max
y2_vector= y2*ones(281)

plot = @pgf Axis(
    {
        xlabel="Matrix dimension [N]",
        ylabel="Operations per second [GFLOPS]",
        title="Experimental vs theorical GLOPS",
        #legend="north east"
    },
    Plot({no_marks, "blue"}, Table(x, y1)),
    Plot({no_marks, "red"}, Table(x, y2_vector)),
)

PGFPlotsX.save("/Users/juanromanbermejo/Desktop/documentacion_CPU/doc_latex/code/3-matmul_vs_theoretical-time/matmul_vs_theoretical-time.tex", plot, include_preamble=false)

#display( plot(N, GFLOPS, ylims=(0, 5000), title="GFLOPS",  minorgrid=true  ) )
#display( plot!(N, GFLOPS_max *ones( length(N) ), minorgrid=true ) )
