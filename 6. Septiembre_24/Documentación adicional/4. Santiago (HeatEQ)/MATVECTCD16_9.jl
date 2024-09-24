using LinearAlgebra, SparseArrays, BenchmarkTools
using Plots

# Problem parameters
nx = 25  # Number of points in x direction
ny = 25  # Number of points in y direction
Lx = 1.0  # Length in x direction
Ly = 1.0  # Length in y direction
alpha = 0.001  # Thermal diffusivity
v = [0, 0.1]  # Velocity vector (vx, vy)
dt = 0.01  # Time step
tfinal = 15  # Final simulation time
nsteps = Int(tfinal / dt)  # Number of time steps

# Grid size
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)

# Function to initialize the temperature field
function initialize_temperature(nx, ny, dx, dy)
    T = zeros(nx, ny)
    for i in 1:nx
        for j in 1:ny
            x = (i-1) * dx
            y = (j-1) * dy
            T[i,j] = exp(-(25*(x-0.5)^2 + 25*(y-0.5)^2))
        end
    end
    return vec(T)
end

# Function to apply boundary conditions
function apply_boundary_conditions!(T_vec, nx, ny)
    T = reshape(T_vec, nx, ny)

    # Apply boundary conditions
    for i in 1:nx
        x = (i-1) * dx
        T[i,1] = exp(-(25*(x-0.5)^2 + 25*(0.0-0.5)^2))
        T[i,end] = exp(-(25*(x-0.5)^2 + 25*(Ly-0.5)^2))
    end
    for j in 1:ny
        y = (j-1) * dy
        T[1,j] = exp(-(25*(0.0-0.5)^2 + 25*(y-0.5)^2))
        T[end,j] = exp(-(25*(Lx-0.5)^2 + 25*(y-0.5)^2))
    end

    return vec(T)
end

# Define nodes in x and y
nodes_x = range(0, stop=Lx, length=nx)
nodes_y = range(0, stop=Ly, length=ny)

# Functions to construct derivative matrices using finite differences
function first_derivative_matrix(n, d)
    e = ones(n)
    D = spdiagm(
        -1 => -1 / (2*d) * e[2:end],
         1 =>  1 / (2*d) * e[1:end-1]
    )
    # Forward difference at the first point
    D[1,1] = -1 / d
    D[1,2] =  1 / d
    # Backward difference at the last point
    D[n,n-1] = -1 / d
    D[n,n]   =  1 / d
    return D
end

function second_derivative_matrix(n, d)
    e = ones(n)
    D2 = spdiagm(
        -1 =>  1 / d^2 * e[2:end],
         0 => -2 / d^2 * e,
         1 =>  1 / d^2 * e[1:end-1]
    )
    # Second derivative at the first and last points (Dirichlet BCs)
    D2[1,1] = 1 / d^2
    D2[1,2] = -2 / d^2
    D2[1,3] = 1 / d^2
    D2[n,n-2] = 1 / d^2
    D2[n,n-1] = -2 / d^2
    D2[n,n]   = 1 / d^2
    return D2
end

# Compute derivative matrices for x and y directions
D_x = first_derivative_matrix(nx, dx)
D_y = first_derivative_matrix(ny, dy)
D2_x = second_derivative_matrix(nx, dx)
D2_y = second_derivative_matrix(ny, dy)

# Construct sparse identity matrices
I_x = spdiagm(0 => ones(nx))
I_y = spdiagm(0 => ones(ny))

# Construct 2D derivative operators using Kronecker products
D_x_2D = kron(I_y, D_x)
D_y_2D = kron(D_y, I_x)
Laplacian = kron(I_y, D2_x) + kron(D2_y, I_x)

# Initialize temperature field as a vector
T_vec = initialize_temperature(nx, ny, dx, dy)
T_new_vec = similar(T_vec)

# List to store temperature states for visualization
temperaturas = []

# Time-stepping loop
elapsedtime = @elapsed begin
    for step in 1:nsteps
        # Compute advection and diffusion terms
        Advection = v[1] * (D_x_2D * T_vec) + v[2] * (D_y_2D * T_vec)
        Diffusion = alpha * (Laplacian * T_vec)
        
        # Update temperature field
        T_new_vec = T_vec .+ dt .* (Diffusion - Advection)
        
        # Apply boundary conditions
        T_new_vec = apply_boundary_conditions!(T_new_vec, nx, ny)
        global T_vec = T_new_vec

        # Save the current temperature state for visualization
        push!(temperaturas, copy(T_vec))
    end
end

# Visualization
x = range(0, stop=Lx, length=nx)
y = range(0, stop=Ly, length=ny)
z_min, z_max = 0, 1.0  # Limits for temperature

# Create 2D contour animation
intervalo_animacion = 30
animation = @animate for i in 1:intervalo_animacion:length(temperaturas)
    t_matrix = reshape(temperaturas[i], nx, ny)
    contourf(x, y, t_matrix, c=:inferno, xlims=(0, Lx), ylims=(0, Ly), zlims=(z_min, z_max),
             xlabel="", ylabel="", title="", clabels=false)
end

# Save animation as a gif
gif(animation, "advection_diffusion_2d.gif", fps=10)
println("Elapsed time for the simulation loop: ", elapsedtime, " seconds")
 