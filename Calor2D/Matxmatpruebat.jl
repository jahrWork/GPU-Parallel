using LinearAlgebra, SparseArrays, Plots

# Parámetros del problema
nx = 50   # Número de puntos en la dirección x
ny = 50   # Número de puntos en la dirección y
Lx = 1.0  # Longitud del dominio en la dirección x
Ly = 1.0  # Longitud del dominio en la dirección y
alpha = 0.01  # Difusividad térmica
dt = 0.001  # Paso de tiempo
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
            if 0.4 < i * dx < 0.6 && 0.4 < j * dy < 0.6
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
    Ix = speye(ny)  # Matriz identidad en y
    Iy = speye(nx)  # Matriz identidad en x
    Dx = spdiagm(0 => -2.0 * ones(nx), 1 => ones(nx - 1), -1 => ones(nx - 1)) / dx^2
    Dy = spdiagm(0 => -2.0 * ones(ny), 1 => ones(ny - 1), -1 => ones(ny - 1)) / dy^2
    Lx = kron(Ix, Dx)
    Ly = kron(Dy, Iy)
    return Lx, Ly
end

# Inicializamos el campo de temperatura
T = initialize_temperature(nx, ny, dx, dy)

# Creamos los operadores Laplacianos 2D
Lx, Ly = create_laplacian_operator(nx, ny, dx, dy)

# Lista para almacenar los estados de la temperatura
temperaturas = []

# Vectorizar la matriz de temperatura
T_vec = vec(T)

# Medir el tiempo de solución solo del bucle
loop_time = @elapsed begin
    for step in 1:nsteps
        T_new_vec = T_vec .+ alpha * dt * (Lx * T_vec + Ly * T_vec)
        T_vec = copy(T_new_vec)
        T_mat = reshape(T_vec, nx, ny)
        apply_boundary_conditions!(T_mat)
        T_vec = vec(T_mat)
        push!(temperaturas, copy(T_mat))  # Guardamos el estado actual de la temperatura
    end
end

println("Tiempo total del bucle: $loop_time segundos")

# Determinamos los límites de los ejes
z_min, z_max = 0, 1.0  # Límites del eje z (temperatura)

# Determinamos los límites de los ejes
x = range(0, stop=Lx, length=nx)
y = range(0, stop=Ly, length=ny)

# Creamos la animación 3D
intervalo_animacion = 25
animation = @animate for i in 1:intervalo_animacion:length(temperaturas)
    t = temperaturas[i]
    surface(x, y, t, title="Distribución de temperatura", xlabel="x", ylabel="y", zlabel="Temperatura", c=:inferno, xlims=(0, Lx), ylims=(0, Ly), zlims=(z_min, z_max))
end

# Guardamos la animación como un gif
gif(animation, "difusion_calor_3d_matmul.gif", fps=30)

println("Tipo de Lx: ", typeof(Lx))
