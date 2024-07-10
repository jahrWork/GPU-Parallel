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
        end
    end
    return A
end

# Inicializamos el campo de temperatura
T = initialize_temperature(nx, ny, dx, dy)

# Creamos el operador Laplaciano 2D
A = create_laplacian_operator(nx, ny, dx, dy, alpha)

# Aplicamos las condiciones de frontera
apply_boundary_conditions!(T, nx, ny)

# Lista para almacenar los estados de la temperatura
temperaturas = []

# Almacenamos el estado inicial
push!(temperaturas, copy(T))

# Resolvemos el sistema de ecuaciones diferenciales
for step in 1:nsteps
    T = T + dt * (A * T)
    apply_boundary_conditions!(T, nx, ny)
    push!(temperaturas, copy(T))
end

# Simulation loop
for step in 1:nsteps
    T = T + dt * (A * T)
    apply_boundary_conditions!(T, nx, ny)
    push!(temperaturas, copy(T))
end

# Graficamos el resultado
heatmap(reshape(T, nx, ny), aspect_ratio=:equal, c=:blues, xlabel="x", ylabel="y", title="Campo de temperatura")

# Graficamos la evolución de la temperatura en el centro del dominio
temperatura_centro = [T[(ny ÷ 2) * nx + nx ÷ 2] for T in temperaturas]
plot(1:length(temperaturas), temperatura_centro, xlabel="Paso de tiempo", ylabel="Temperatura", title="Evolución de la temperatura en el centro del dominio")

# Guardamos la animación de la evolución de la temperatura
anim = @animate for T in temperaturas
    heatmap(reshape(T, nx, ny), aspect_ratio=:equal, c=:blues, xlabel="x", ylabel="y", title="Campo de temperatura")
end

gif(anim, "evolucion_temperatura.gif", fps=30)
# Benchmark
@benchmark begin
    T = initialize_temperature(nx, ny, dx, dy)
    A = create_laplacian_operator(nx, ny, dx, dy, alpha)
    apply_boundary_conditions!(T, nx, ny)
    temperaturas = []
    push!(temperaturas, copy(T))
    for n in 1:nsteps
        T = T + dt * (A * T)
        apply_boundary_conditions!(T, nx, ny)
        push!(temperaturas, copy(T))
    end
end
