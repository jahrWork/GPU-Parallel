using BenchmarkTools
using Plots


function measure_gflops(size)
    a = rand(Float64, size)
    b = rand(Float64, size)

  
    time = @belapsed $a + $b

   
    flops = 2 * size
    gflops = flops / (time * 1e9)

    return gflops
end


sizes = [10^i for i in 1:6]
gflops_results = [measure_gflops(size) for size in sizes]


plot(sizes, gflops_results, xlab="Vector Size", ylab="GFLOPS", title="GFLOPS vs Vector Size (suma)", lw=2, marker=:o)
