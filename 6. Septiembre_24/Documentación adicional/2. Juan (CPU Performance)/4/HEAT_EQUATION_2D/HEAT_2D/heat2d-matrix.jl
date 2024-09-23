using LinearAlgebra, SparseArrays, Plots

# Parámetros del problema
nx = 100   # Número de puntos en la dirección x
ny = 100   # Número de puntos en la dirección y
Lx = 2.0  # Longitud del dominio en la dirección x
Ly = 2.0  # Longitud del dominio en la dirección y
alpha = 1  # Difusividad térmica
dt = 0.1  # Paso de tiempo
tfinal = 1  # Tiempo final de la simulación
nsteps = Int(tfinal / dt)  # Número de pasos de tiempo

# Definimos el tamaño de la malla
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)

# Función para inicializar el campo de temperatura
function initialize_temperature(nx, ny, dx, dy)
    T = zeros(nx, ny)
    for i in 1:nx
        for j in 1:ny
            if 0.4 < i*dx < 0.6 && 0.4 < j*dy < 0.6
                T[i, j] = 1.0
            end
        end
    end
    return T
end

# Función para aplicar las condiciones de frontera (Dirichlet en este caso)
function apply_boundary_conditions!(T)
    T[1, :] .= 0.0
    T[end, :] .= 0.0
    T[:, 1] .= 0.0
    T[:, end] .= 0.0
end

# Función para crear el operador Laplaciano 2D de forma matricial
function create_laplacian_operator(nx, ny, dx, dy)
    Lx = spdiagm(0 => -2.0*ones(nx), 1 => ones(nx-1), -1 => ones(nx-1)) / dx^2
    Ly = spdiagm(0 => -2.0*ones(ny), 1 => ones(ny-1), -1 => ones(ny-1)) / dy^2
    return Lx, Ly
end

# Función para ejecutar la simulación
function simulate_heat_diffusion(nx, ny, Lx, Ly, alpha, dt, nsteps)
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)

    # Inicializamos el campo de temperatura
    T = initialize_temperature(nx, ny, dx, dy)
    T_new = similar(T)

    # Creamos los operadores Laplacianos 2D
    Lx_op, Ly_op = create_laplacian_operator(nx, ny, dx, dy)

    # Ejecutamos la simulación
    for step in 1:nsteps
        T_new .= T .+ alpha * dt * (Lx_op * T + T * Ly_op')
        T, T_new = T_new, T
        apply_boundary_conditions!(T)
    end

    return T
end

# Medimos el tiempo usando @time (similar al primer código)
@time begin
    final_temperature = simulate_heat_diffusion(nx, ny, Lx, Ly, alpha, dt, nsteps)
end
