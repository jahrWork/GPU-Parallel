using CPUTime
using Plots

using LinearAlgebra, MKL

function vector_multiplication2(A,B)

    return @elapsed dot(A, B)  
  
end

#declaracion de variables del bucle
NTOTAL = 1000000
PASO = 1000
INICIO = 1000
ESPACIO = div((NTOTAL-INICIO),PASO) + 1

#declaracion de vectores para la representacion
vectorAxB = zeros(Float32,ESPACIO,1)
infoN = zeros(Float32,ESPACIO,1)
for N in INICIO:PASO:NTOTAL
    A = rand(Float32, N)  
    B = rand(Float32, N)  

    #dot(AXB)
    time_dot= vector_multiplication2(A,B)
    GFLOPS_dot = 1e-9 / (time_dot/(2*N))

    i = div(N, PASO)
    vectorAxB[i] = GFLOPS_dot  
    infoN[i] = N
end 

plot(infoN, vectorAxB, title="Resultado Variando N", label="dot(A,B)", xlabel="N", ylabel="GFLOPS")
