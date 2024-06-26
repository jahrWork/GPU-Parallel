using LinearAlgebra
using Plots
using MKL

a=rand(3)
# Función para calcular gigaflops
function calcula_gigaflops(n)
    A = rand(n)
    B = rand(n)
    
    tiempo = @elapsed dot(A, B)
    
    # Número de operaciones de punto flotante: 2n (n multiplicaciones, n adiciones)
    flops = 2n
    
    gigaflops = flops / (tiempo * 1e9)
    return gigaflops
end

# Tamaño de vectores a testear
tamaños = [100, 1000, 10000, 100000, 1000000]

# Calcula gigaflops para cada tamaño
gigaflops = [calcula_gigaflops(n) for n in tamaños]

# Gráfico
plot(tamaños, gigaflops, title="Rendimiento de Producto Escalar en Gigaflops",
     xlabel="Tamaño del Vector", ylabel="Gigaflops", xscale=:log10, yscale=:log10, legend=false)


@code_native dot(a,a)