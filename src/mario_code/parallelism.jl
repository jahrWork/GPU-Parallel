import Pkg
Pkg.activate(".") 
Pkg.add( "Plots" ) 
Pkg.add( "CPUTime" ) 
Pkg.add( "MKL" )
Pkg.add("CUDA")

using CPUTime
using Plots
using LinearAlgebra
using CUDA
using MKL

# Scalar product of a matrix and a vector
function my_matrix_vector_multiplication(A,B)
  (N, M) = size(A)
  C = zeros(Float32,  (N) )
  for i in 1:N
    for k in 1:M
      C[i] = C[i] + A[i,k]*B[k]
    end  
  end
  return C 
end

function my_efficient_matrix_vector_multiplication(A,B)
  (N, M) = size(A)
  BT = transpose(B) 
  C = zeros(Float32,  (N) )
  for k in 1:M
    for i in 1:N
      C[i] = C[i] + A[i,k]*BT[k]
    end  
  end
  return C 
end

# Matrix product
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

# Matrix product with CUDA
function my_efficient_matrix_multiplication_cuda(A,B)
  (N, M) = size(A)
  (M, L) = size(B) 
  BT = transpose(B) 
  C = CUDA.zeros(Float32,  (N, L) )
  for k in 1:M
    for j in 1:L, i in 1:N
      C[i,j] = C[i,j] + A[i,k]*BT[j,k]
    end  
  end
  return C 
end

# Matrix product with MKL
function my_efficient_matrix_multiplication_mkl(A,B)
  (N, M) = size(A)
  (M, L) = size(B) 
  BT = transpose(B) 
  C = MKL.zeros(Float32,  (N, L) )
  for k in 1:M
    for j in 1:L, i in 1:N
      C[i,j] = C[i,j] + A[i,k]*BT[j,k]
    end  
  end
  return C 
end

# Scalar product of a matrix using MKL
function my_matrix_vector_multiplication_mkl(A,B)
  (N, M) = size(A)
  C = MKL.zeros(Float32,  (N) )
  for i in 1:N
    for k in 1:M
      C[i] = C[i] + A[i,k]*B[k]
    end  
  end
  return C 
end

# Scalar product of a matrix using CUDA
function my_matrix_vector_multiplication_cuda(A,B)
  (N, M) = size(A)
  C = CUDA.zeros(Float32,  (N) )
  for i in 1:N
    for k in 1:M
      C[i] = C[i] + A[i,k]*B[k]
    end  
  end
  return C 
end

# Time measurement
function time_measurement(f, A, B)
  t = CPUTime.CPUTime()
  CPUTime.start!(t)
  f(A,B)
  CPUTime.stop!(t)
  return CPUTime.elapsed!(t)
end

# Time measurement for CUDA
function time_measurement_cuda(f, A, B)
  t = CPUTime.CPUTime()
  CPUTime.start!(t)
  CUDA.@sync f(A,B)
  CPUTime.stop!(t)
  return CPUTime.elapsed!(t)
end

# Time measurement for MKL
function time_measurement_mkl(f, A, B)
  t = CPUTime.CPUTime()
  CPUTime.start!(t)
  MKL.@sync f(A,B)
  CPUTime.stop!(t)
  return CPUTime.elapsed!(t)
end

# Time measurement for a matrix and a vector
function time_measurement_vector(f, A, B)
  t = CPUTime.CPUTime()
  CPUTime.start!(t)
  f(A,B)
  CPUTime.stop!(t)
  return CPUTime.elapsed!(t)
end

# Time measurement for a matrix and a vector using CUDA
function time_measurement_vector_cuda(f, A, B)
  t = CPUTime.CPUTime()
  CPUTime.start!(t)
  CUDA.@sync f(A,B)
  CPUTime.stop!(t)
  return CPUTime.elapsed!(t)
end

# Time measurement for a matrix and a vector using MKL
function time_measurement_vector_mkl(f, A, B)
  t = CPUTime.CPUTime()
  CPUTime.start!(t)
  MKL.@sync f(A,B)
  CPUTime.stop!(t)
  return CPUTime.elapsed!(t)
end

# Plotting
function plot_results(N, times, times_cuda, times_mkl, title)
  plot(N, times, label="CPU", xlabel="Matrix size", ylabel="Time (s)", title=title)
  plot!(N, times_cuda, label="CUDA")
  plot!(N, times_mkl, label="MKL")
end

# Main function
function main()
  N = [10, 100, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
  times = []
  times_cuda = []
  times_mkl = []
  for n in N
    A = rand(Float32, (n,n))
    B = rand(Float32, (n,n))
    push!(times, time_measurement(my_efficient_matrix_multiplication, A, B))
    push!(times_cuda, time_measurement_cuda(my_efficient_matrix_multiplication_cuda, A, B))
    push!(times_mkl, time_measurement_mkl(my_efficient_matrix_multiplication_mkl, A, B))
  end
  plot_results(N, times, times_cuda, times_mkl, "Matrix multiplication")
end

main()