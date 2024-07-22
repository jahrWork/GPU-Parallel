using Plots
using Printf

# Parameters
nx, ny = 100, 100       # Number of grid points in x and y directions
Lx, Ly = 1.0, 1.0       # Physical dimensions of the domain
dx, dy = Lx/nx, Ly/ny   # Grid spacing
alpha = 0.01            # Thermal diffusivity
T_inf = 20.0            # Temperature of the inflow air
T_rect = 100.0          # Temperature of the rectangle
u = 0.1                 # Flow velocity (left to right)

# Stability check
dt_max = min(0.5 * (dx^2 * dy^2) / (alpha * (dx^2 + dy^2)), dx / u)
dt = min(dt_max, 0.0001)  # Choose a dt smaller or equal to the stable dt_max

println("Time step chosen: $dt")
println("Maximum stable time step: $dt_max")

# Calculate number of time steps
t_end = 1.0          # End time
nsteps = Int(t_end / dt)

# Initialize temperature field
T = fill(T_inf, nx+1, ny+1)

# Indices of the rectangle
rect_x = Int(floor(nx/8)):Int(floor(3*nx/8))
rect_y = Int(floor(ny/8)):Int(floor(3*ny/8))
T[rect_x, rect_y] .= T_rect

# Function to apply boundary conditions
function apply_boundary_conditions!(T, T_inf, u, dt, dx)
    nx, ny = size(T)
    
    # Inflow boundary (left side)
    T[1, :] .= T_inf
    
    # Outflow boundary (right side) - Convective outflow boundary condition
    for j in 1:ny
        T[end, j] = T[end, j] - u * dt / dx * (T[end, j] - T[end-1, j])
    end
    
    # Top boundary - Neumann boundary condition
    T[:, end] .= T[:, end-1]
    
    # Bottom boundary - Neumann boundary condition
    T[:, 1] .= T[:, 2]
end

# Function to update the temperature field
function update_temperature!(T, alpha, u, dx, dy, dt)
    nx, ny = size(T)
    T_new = copy(T)
    for i in 2:nx-1
        for j in 2:ny-1
            # Convection term
            convection = -u * (T[i,j] - T[i-1,j]) / dx
            
            # Diffusion term
            diffusion = alpha * (
                (T[i+1,j] - 2*T[i,j] + T[i-1,j]) / dx^2 +
                (T[i,j+1] - 2*T[i,j] + T[i,j-1]) / dy^2 )
            
            # Update temperature
            T_new[i,j] = T[i,j] + dt * (convection + diffusion)
        end
    end
    return T_new
end

# Function to set the rectangle temperature
function set_rectangle_temperature!(T, rect_x, rect_y, T_rect)
    T[rect_x, rect_y] .= T_rect
end

# Visualization setup
anim = Animation()

# Time integration loop
for step in 1:nsteps
    # Apply boundary conditions
    apply_boundary_conditions!(T, T_inf, u, dt, dx)

    # Update temperature field
  global  T = update_temperature!(T, alpha, u, dx, dy, dt)

    # Set rectangle temperature
    set_rectangle_temperature!(T, rect_x, rect_y, T_rect)

    # Visualization
    if step % 10 == 0
        heatmap(T', c=:viridis, clim=(T_inf, T_rect), title=@sprintf("Time: %.3f s", step*dt))
        frame(anim)
    end
end

# Save the animation as a gif
gif(anim, "heat_transfer_simulation1.gif", fps=10)
