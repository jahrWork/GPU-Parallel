import Pkg
Pkg.activate(".")  # environment in this folder
Pkg.add( "Plots" ) 
Pkg.add( "CPUTime" ) 
#Pkg.add( "BLIS" )
Pkg.add( "MKL" )
Pkg.add( "PGFPlotsX" )
using CPUTime
using Plots
using LinearAlgebra
using PGFPlotsX


# Function to initialize two random matrices A and B of size N x N
function matrix_initialization(N)


  A = rand(Float32, N, N )
  B = rand(Float32, N, N )

  return A, B 

end 


# Function to perform matrix multiplication without timing (basic operation)
function matrix_multiplication(A,B)

  return  A * B  

end


# Function to time the matrix multiplication process and calculate time per operation
function time_matrix_multilication(N, N_cores, matinit, matmul)

  Time = zeros( length(N) )
  #Theoretical_time = 4e9/(4e9 * 512/32 * 4 * N_cores)
  Theoretical_time = 1e9/(4e9 * 512/32 *  N_cores) # jahr

  for (i,n) in enumerate(N)  # variables inside loop have local scope 
 
   A,B = matinit(n)
   m = length(B)

   #  dt = 1e9 * matmul(A,B)
 
   t1 = time_ns()
   matmul(A,B)
   t2 = time_ns()
   dt = t2-t1
   
   Time[i] = dt/(2*n*m)
   

   println("N=", n, " Time per operation =", Time[i] , " nsec")
   println("N=", n, " Theoretical time per operation =", Theoretical_time, " nsec")
    
  end 
  
  return Time, Theoretical_time

end 










 
function plot_combined(N_threads_range)
  # Initialize the vectors to store GFLOPS for each thread count
  y1 = Float32[]
  y2 = Float32[]
  y3 = Float32[]
  y4 = Float32[]
  N = Vector{Int}[]  # To store the values of N (matrix dimensions)
  
  # Loop over the number of threads (N_threads) from 1 to 4
  for N_threads in N_threads_range
      N_threads = N_threads 
      N_cores = N_threads 
      println("Threads =", N_threads ) 
      println("Cores =", N_cores ) 
      
      # Define the matrix size range N
      N_matrix = Vector([10:10:2500; 2500:100:5000])
      BLAS.set_num_threads(N_cores)  # Set the number of threads for BLAS operations
      println(" threads = ", BLAS.get_num_threads(), " N_cores =", N_cores )
      
      # Measure the time for matrix multiplication for each size N
      Time, Theoretical_time = time_matrix_multilication(N_matrix, N_cores, matrix_initialization, matrix_multiplication)
      GFLOPS = 1 ./ Time  # Calculate GFLOPS (Giga Floating Point Operations Per Second)
      
      # Store the GFLOPS results in the respective vectors
      if N_threads == 1
          y1 = GFLOPS  # Store in y1 if N_threads is 1
      elseif N_threads == 2
          y2 = GFLOPS  # Store in y2 if N_threads is 2
      elseif N_threads == 3
          y3 = GFLOPS  # Store in y3 if N_threads is 3
      elseif N_threads == 4
          y4 = GFLOPS  # Store in y4 if N_threads is 4
      end
      
      # Ensure the N vector is stored only once
      if isempty(N)
          N = N_matrix
      end
  end
  
  # Return the data: N (matrix dimensions), y1, y2, y3, y4
  return N, y1, y2, y3, y4
end

# Extract the GFLOPS data for plotting
N, y1, y2, y3, y4 = plot_combined(1:4)

# Create the plot using PGFPlotsX
plot = @pgf Axis(
    {
        xlabel="Matrix dimension [N]",
        ylabel="GFLOPS",
        title="Comparison of different dot functions",
        legend_pos="north west",  # Position of the legend
        legend_entries={"Threads = 1", "Threads = 2", "Threads = 3", "Threads = 4"},  # Add legend entries here
        ymin=0,  # Set the minimum value for the y-axis
        ymax=400  # Set the maximum value for the y-axis (increase height)
    },
    Plot({no_marks, "blue"}, Table(N, y1)),  # Plot for Threads = 1
    Plot({no_marks, "red"}, Table(N, y2)),   # Plot for Threads = 2
    Plot({no_marks, "green"}, Table(N, y3)), # Plot for Threads = 3
    Plot({no_marks, "black"}, Table(N, y4))  # Plot for Threads = 4
)

# Save the plot in .tex format
PGFPlotsX.save("/Users/juanromanbermejo/Desktop/documentacion_CPU/doc_latex/code/2-singlecore_vs_multicore/n_threads.tex", plot, include_preamble=false)
