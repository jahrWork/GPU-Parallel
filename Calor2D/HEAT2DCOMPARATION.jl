using SparseArrays
using LinearAlgebra
using BenchmarkTools
using Plots

# Parámetros del problema
nx = 100
ny = 100
Lx = 1.0
Ly = 1.0
alpha = 0.01
dt = 0.001
tfinal = 1.0
nsteps = Int(tfinal / dt)

dx = Lx / (nx - 1)
dy = Ly / (ny - 1)

# Inicialización de la temperatura
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

# Aplicar condiciones de frontera
function apply_boundary_conditions!(T, nx, ny)
    for i in 1:nx
        T[i] = 0.0
        T[(ny - 1) * nx + i] = 0.0
    end
    for j in 1:ny
        T[(j - 1) * nx + 1] = 0.0
        T[(j - 1) * nx + nx] = 0.0
    end
end

# Crear el operador Laplaciano
function create_laplacian_operator(nx, ny, dx, dy, alpha)
    N = nx * ny
    A = spzeros(N, N)
    for j in 1:ny
        for i in 1:nx
            k = (j - 1) * nx + i
            if i > 1
                A[k, k - 1] = alpha * dt / dx^2
            end
            if i < nx
                A[k, k + 1] = alpha * dt / dx^2
            end
            if j > 1
                A[k, k - nx] = alpha * dt / dy^2
            end
            if j < ny
                A[k, k + nx] = alpha * dt / dy^2
            end
            A[k, k] = 1.0 - 2.0 * alpha * dt * (1.0 / dx^2 + 1.0 / dy^2)
        end
    end
    return A
end

# Método matriz por vector
function heat2d_matvec!(T, A, nx, ny, dt)
    T .= A * T
    apply_boundary_conditions!(T, nx, ny)
    return T
end

# Método matriz por matriz
function heat2d_matmat!(T, A, nx, ny, dt)
    T .= A * T
    apply_boundary_conditions!(T, nx, ny)
    return T
end

# Inicialización
T = initialize_temperature(nx, ny, dx, dy)
apply_boundary_conditions!(T, nx, ny)
A = create_laplacian_operator(nx, ny, dx, dy, alpha)

# Benchmark de matriz por vector
@btime heat2d_matvec!(T, A, nx, ny, dt)

# Inicialización de nuevo para asegurar estado limpio
T = initialize_temperature(nx, ny, dx, dy)
apply_boundary_conditions!(T, nx, ny)

# Benchmark de matriz por matriz
@btime heat2d_matmat!(T, A, nx, ny, dt)

# Visualización de resultados
function plot_temperature(T, nx, ny)
    x = range(0, stop=1, length=nx)
    y = range(0, stop=1, length=ny)
    heatmap(x, y, reshape(T, nx, ny), c=:blues, xlabel="x", ylabel="y", aspect_ratio=:equal)
end

temperaturas = []

T = initialize_temperature(nx, ny, dx, dy)
apply_boundary_conditions!(T, nx, ny)
A = create_laplacian_operator(nx, ny, dx, dy, alpha)

push!(temperaturas, copy(T))

for step in 1:nsteps
    Tmp = heat2d_matvec!(T, A, nx, ny, dt)
    push!(temperaturas, copy(T))
end

plot_temperature(T, nx, ny)

# Guardar la animación como GIF
@gif for T in temperaturas
    plot_temperature(T, nx, ny)
end every 10

# Benchmark con bucles completos para matriz por vector
@benchmark begin
    T = initialize_temperature(nx, ny, dx, dy)
    apply_boundary_conditions!(T, nx, ny)
    temperaturas = []
    push!(temperaturas, copy(T))
    for n in 1:nsteps
        T = heat2d_matvec!(T, A, nx, ny, dt)
        push!(temperaturas, copy(T))
    end
end

# Benchmark con bucles completos para matriz por matriz
@benchmark begin
    T = initialize_temperature(nx, ny, dx, dy)
    apply_boundary_conditions!(T, nx, ny)
    temperaturas = []
    push!(temperaturas, copy(T))
    for n in 1:nsteps
        T = heat2d_matmat!(T, A, nx, ny, dt)
        push!(temperaturas, copy(T))
    end
end
