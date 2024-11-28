
using CPUTime, Plots, LinearAlgebra, MKL, BenchmarkTools


function matmul_1t(Nt, A, B, C)
  
    BLAS.set_num_threads(1)
    Threads.@threads for k in 1:Nt 
        C =  k * A * B 
    end
  
    return C 
  end 

function init(N)

  A = rand(Float32, N, N)
  B = rand(Float32, N, N)
  Nop = 2 * N^3
  return A, B, Nop 

end



function plot_GFLOPS()


 
  N =  Vector( [400:25:1000;] )

  GFLOPS = zeros(size(N))


  for (i, n) in enumerate(N) 
       
        A, B, Nop = init(n)
        C = zeros(Float32, n, n)
        Nt = 100 
        Nop = Nop * Nt

		dt = @belapsed matmul_1t($Nt, $A, $B, $C)
		
		Time = (dt / Nop) * 1e9  # Convert seconds to nanoseconds
        GFLOPS[i] = 1 / Time 
        println( "N =", n, " GFLOPS = ", GFLOPS[i] )
  end     
  
  plot(N,  GFLOPS, 
       title = "GFLOPS matmul versus N", 
       xlabel = "\$ N \$", ylabel = "\$ GFLOPS \$", 
       ylimits=(0,maximum(GFLOPS)+100)
      )
  savefig("plot_GFLOPS_R5_5625u_laptop.png")

end 

plot_GFLOPS()