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
    V = rand(Float32, N, 1) 
    return @elapsed A * V
end

#declaracion de variables del bucle
NTOTAL = 50000
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

    #inicializacion
    A, B = matrix_initialization(N)

    #A X VECTORES 2
    time_AxV2= multiply_matrix_by_n_vectors222(A,N)
    GFLOPS_AxV2 = 1e-9 / (time_AxV2/(2*N^2))

    i = div(N, PASO)
    vectorAxV2[i] = GFLOPS_AxV2
    infoN[i] = N
end 

plot(infoN, vectorAxV2, title="Resultado Mat x Vec", label="matrizxmatriz", xlabel="N", ylabel="GFLOPS")
