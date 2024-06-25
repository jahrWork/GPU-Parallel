using DifferentialEquations
using SparseArrays
using Plots

# Definición de parámetros
Lx = 1.0              # Longitud en x
Ly = 1.0              # Longitud en y
Nx = 50               # Número de puntos en x
Ny = 50               # Número de puntos en y
dx = Lx / (Nx - 1)    # Espaciado en x
dy = Ly / (Ny - 1)    # Espaciado en y
alpha = 0.01          # Difusividad térmica
dt = 0.001             # Paso temporal
Nt = 1000             # Número de pasos temporales

# Construir la matriz A
N = Nx * Ny
A = spzeros(N, N)

function index(i, j, Nx, Ny)
    return (j-1) * Nx + i
end

for j in 1:Ny
    for i in 1:Nx
        k = index(i, j, Nx, Ny)
        if i > 1
            A[k, index(i-1, j, Nx, Ny)] = alpha / dx^2
        end
        if i < Nx
            A[k, index(i+1, j, Nx, Ny)] = alpha / dx^2
        end
        if j > 1
            A[k, index(i, j-1, Nx, Ny)] = alpha / dy^2
        end
        if j < Ny
            A[k, index(i, j+1, Nx, Ny)] = alpha / dy^2
        end
        A[k, k] = -2 * alpha * (1/dx^2 + 1/dy^2)
    end
end

# Inicialización de la temperatura
u = zeros(Nx, Ny)

# Condiciones iniciales en los bordes
for i in 1:Nx
    u[i, 1] = 100.0  # borde inferior
    u[i, Ny] = 100.0 # borde superior
end

for j in 1:Ny
    u[1, j] = 100.0  # borde izquierdo
    u[Nx, j] = 100.0 # borde derecho
end

# Vectorizar u
u_vec = vec(u)

# función que calcula du/dt usando la matriz A
function heat_equation!(du, u, p, t)
    du .= A * u
end

# Preparar el problema y parámetros
tspan = (0.0, Nt * dt)
prob = ODEProblem(heat_equation!, u_vec, tspan)

# Solución usando Runge-Kutta 4
sol = solve(prob, RK4(), dt=dt)

# Solución final puesta en forma de matriz
u_final_vec = sol[end]
u_final = reshape(u_final_vec, Nx, Ny)

# distribución final de temperatura
x = range(0, stop=Lx, length=Nx)
y = range(0, stop=Ly, length=Ny)
heatmap(x, y, u_final, title="Distribución final de temperatura", xlabel="x", ylabel="y", colorbar_title="Temperatura")
