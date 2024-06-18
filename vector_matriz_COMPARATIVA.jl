using CPUTime
using Plots

#using LinearAlgebra, BLIS
using LinearAlgebra, MKL
#using LinearAlgebra
#using CUDA

function matrix_initialization(N)


  A = rand(Float32, N, N )
  B = rand(Float32, N, N )

  return A, B 

end 

function matrix_multiplication(A,B)

    return @elapsed A * B  
 
end

function matrix_add2(A,B)

    return @elapsed A + B  
  
end

function vector_multiplication2(A,B)

    return @elapsed dot(A, B)  
  
end

function multiply_matrix_by_n_vectors(A, V, N)

    a = @elapsed begin
    for i in 1:N
        b = A * V[:, i] 
    end
    end
    return a

end

function multiply_matrix_by_n_vectors222(A,N)
    a = @elapsed begin
    for i in 1:N
        V = rand(Float32, N, 1) 
        A * V
    end
    end
    return a
end

#declaracion de variables del bucle
NTOTAL = 4000
PASO = 100
INICIO = 100
ESPACIO = div((NTOTAL-INICIO),PASO) + 1

#declaracion de vectores para la representacion
vectorAxB = zeros(Float32,ESPACIO,1)
vectorAxV = zeros(Float32,ESPACIO,1)
vectorAxV2 = zeros(Float32,ESPACIO,1)
infoN = zeros(Float32,ESPACIO,1)

for N in INICIO:PASO:NTOTAL
    A = rand(Float32, N, N)  
    B = rand(Float32, N, N)  
    V = rand(Float32, N, N) 

    #inicializacion
    A, B = matrix_initialization(N)

    #AXB
    time_AxB= matrix_multiplication(A,B)
    GFLOPS_AxC = 1e-9 / (time_AxB/(2*N^3))

    #A X VECTORES EN MATRIZ
    time_AxV= multiply_matrix_by_n_vectors(A,V,N)
    GFLOPS_AxV = 1e-9 / (time_AxV/(2*N^3))

    #A X VECTORES 2
    time_AxV2= multiply_matrix_by_n_vectors222(A,N)
    GFLOPS_AxV2 = 1e-9 / (time_AxV2/(2*N^3))

    i = div(N, PASO)
    vectorAxB[i] = GFLOPS_AxC  
    vectorAxV[i] = GFLOPS_AxV
    vectorAxV2[i] = GFLOPS_AxV2
    infoN[i] = N
end 

plot(infoN, vectorAxB, title="Resultado Variando N", label="matrizxmatriz", xlabel="N", ylabel="GFLOPS")
plot!(infoN, vectorAxV, label="matrizxvector1")
plot!(infoN, vectorAxV2, label="matrizxvector2")