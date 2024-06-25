using LinearAlgebra, SparseArrays, Plots
using BenchmarkTools

# Parámetros del problema
nx = 100  # Número de puntos en la dirección x
ny = 100   # Número de puntos en la dirección y
Lx = 1.0  # Longitud del dominio en la dirección x
Ly = 1.0  # Longitud del dominio en la dirección y
alpha = 0.01  # Difusividad térmica
dt = 0.001  # Paso de tiempo
tfinal = 1.0  # Tiempo final de la simulación
nsteps = Int(tfinal / dt)  # Número de pasos de tiempo

# Definimos el tamaño de la malla
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)

# Función para inicializar el campo de temperatura
function initialize_temperature(nx, ny, dx, dy)
    T = zeros(nx * ny)
    for i in 1:nx
        for j in 1:ny
            if 0.4 < (i-1) * dx < 0.6 && 0.4 < (j-1) * dy < 0.6
                T[(j - 1) * nx + i] = 1.0
            end
        end
    end
    return T
end

# Función para aplicar las condiciones de frontera (Dirichlet en este caso)
function apply_boundary_conditions!(T, nx, ny)
    for i in 1:nx
        T[i] = 0.0          # borde inferior
        T[(ny - 1) * nx + i] = 0.0  # borde superior
    end
    for j in 1:ny
        T[(j - 1) * nx + 1] = 0.0  # borde izquierdo
        T[(j - 1) * nx + nx] = 0.0 # borde derecho
    end
end

# Función para crear el operador Laplaciano 2D de forma matricial
function create_laplacian_operator(nx, ny, dx, dy, alpha)
    N = nx * ny
    A = spzeros(N, N)
    for j in 1:ny
        for i in 1:nx
            k = (j - 1) * nx + i
            if i > 1
                A[k, k - 1] = alpha / dx^2
            end
            if i < nx
                A[k, k + 1] = alpha / dx^2
            end
            if j > 1
                A[k, k - nx] = alpha / dy^2
            end
            if j < ny
                A[k, k + nx] = alpha / dy^2
            end
            A[k, k] = -2 * alpha * (1 / dx^2 + 1 / dy^2)
        end
    end
    return A
end

# Inicializamos el campo de temperatura
T = initialize_temperature(nx, ny, dx, dy)

# Creamos el operador Laplaciano 2D
A = create_laplacian_operator(nx, ny, dx, dy, alpha)

# Lista para almacenar los estados de la temperatura
temperaturas = []

# Ejecutamos la simulación y medimos el tiempo solo del bucle
loop_time = @elapsed begin
    local T_new = copy(T)
    for step in 1:nsteps
        T_new .= T .+ dt * (A * T)
        apply_boundary_conditions!(T_new, nx, ny)
        T .= T_new
        push!(temperaturas, copy(T))  # Guardamos el estado actual de la temperatura
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
    t = reshape(temperaturas[i], nx, ny)
    surface(x, y, t, title="Distribución de temperatura", xlabel="x", ylabel="y", zlabel="Temperatura", c=:inferno, xlims=(0, Lx), ylims=(0, Ly), zlims=(z_min, z_max))
end

# Guardamos la animación como un gif
gif(animation, "difusion_calor_3d_matvec.gif", fps=30)
