import Pkg
Pkg.activate(".")  # environment in this folder
Pkg.add( "Plots" ) 
Pkg.add( "CPUTime" ) 
#Pkg.add( "BLIS" )
Pkg.add( "MKL" )
#Pkg.precompile()

#using LinearAlgebra, BLIS
using LinearAlgebra, MKL
using Plots

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
    Theoretical_time = 4e9/(4e9 * 512/32 * 4 * N_cores)
  
    for (i,n) in enumerate(N)  
   
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

N_threads = Threads.nthreads()
N_cores =  N_threads/2
println("Threads =", N_threads ) 
println("Cores =", N_cores ) 

time_matrix_multilication(2000, N_cores, matrix_multiplication) 

N = Vector([10:10:2500; 2500:100:10000])

Time, Theoretical_time = time_matrix_multilication(N, N_cores, matrix_multiplication)
GFLOPS = 1 ./ Time
GFLOPS_max = 1 / Theoretical_time





plt_time = plot(N, Time, label="Tiempo por operación", yscale=:log10, xlabel="Tamaño de la matriz N", ylabel="Tiempo (nsec)", title="Rendimiento de Multiplicación de Matrices")
plot!(plt_time, N, fill(Theoretical_time, length(N)), label="Tiempo teórico por operación", linestyle=:dash)


plt_gflops = plot(N, GFLOPS, label="GFLOPS", xlabel="Tamaño de la matriz N", ylabel="GFLOPS", title="GFLOPS vs Tamaño de la Matriz")
plot!(plt_gflops, N, fill(GFLOPS_max, length(N)), label="GFLOPS Máximo Teórico", linestyle=:dash)

display(plt_time)
display(plt_gflops)

savefig(plt_time, "tiempo_por_ope1racion2
.png")
savefig(plt_gflops, "gflops_vs_tamano1_matriz2.png")