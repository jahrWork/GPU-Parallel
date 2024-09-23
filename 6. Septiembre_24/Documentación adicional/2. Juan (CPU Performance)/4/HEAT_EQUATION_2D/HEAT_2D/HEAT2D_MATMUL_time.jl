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

# Inicializamos el campo de temperatura
T = initialize_temperature(nx, ny, dx, dy)
T_new = similar(T)

# Creamos los operadores Laplacianos 2D
Lx, Ly = create_laplacian_operator(nx, ny, dx, dy)

# Lista para almacenar los estados de la temperatura
temperaturas = []

#Medimos el tiempo de ejecucion
@elapsed begin
    # Ejecutamos la simulación
    for step in 1:nsteps
        T_new .= T .+ alpha * dt * (Lx * T + T * Ly')
        T, T_new = T_new, T
        apply_boundary_conditions!(T)
        push!(temperaturas, copy(T))  # Guardamos el estado actual de la temperatura
    end
end



# Determinamos los límites de los ejes
#  x = range(0, stop=Lx, length=nx)
#  y = range(0, stop=Ly, length=ny)
#  z_min, z_max = 0, 1.0  # Límites del eje z (temperatura)

 # Determinamos los límites de los ejes
 x = range(0, stop=1, length=nx)
 y = range(0, stop=1, length=ny)


 # Creamos la animación 3D
 intervalo_animacion = 25
 animation = @animate for i in 1:intervalo_animacion:length(temperaturas)
     t = temperaturas[i]
     surface(x, y, reshape(t, nx, ny), title="Distribución de temperatura", xlabel="x", ylabel="y", zlabel="Temperatura", c=:inferno, xlims=(0, Lx), ylims=(0, Ly), zlims=(z_min, z_max))
 end

 # Guardamos la animación como un gif
 gif(animation, "difusion_calor_3d_matmul.gif", fps=30)


 typeof(Lx)