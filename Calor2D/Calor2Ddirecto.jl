using DifferentialEquations
using Plots

# Definición de parámetros
Lx = 1.0              # Longitud en x
Ly = 1.0              # Longitud en y
Nx = 50             # Número de puntos en x
Ny = 100             # Número de puntos en y
dx = Lx / (Nx - 1)    # Espaciado en x
dy = Ly / (Ny - 1)    # Espaciado en y
alpha = 0.01          # Difusividad térmica
dt = 0.01             # Paso temporal
Nt = 1000             # Número de pasos temporales

# Inicialización de la temperatura
u = zeros(Nx, Ny)

# Condiciones iniciales en los bordes
for i in 1:Nx
    u[i, 1] = 20.0  # borde inferior
    u[i, Ny] = 100.0 # borde superior
end

for j in 1:Ny
    u[1, j] = 100.0  # borde izquierdo
    u[Nx, j] = 100.0 # borde derecho
end

# Función para calcular el Laplaciano (discretización de diferencias finitas)
function laplacian(u, i, j, dx, dy)
    return (u[i+1, j] - 2*u[i, j] + u[i-1, j]) / dx^2 + (u[i, j+1] - 2*u[i, j] + u[i, j-1]) / dy^2
end

# Función que define la ecuación del calor en 2D
function heat_equation!(du, u, p, t)
    Nx, Ny, dx, dy, alpha = p
    for i in 2:Nx-1
        for j in 2:Ny-1
            du[i, j] = alpha * laplacian(u, i, j, dx, dy)
        end
    end

    # Aplicar condiciones de flujo (Neumann) en los bordes
    for i in 1:Nx
        du[i, 1] = 0.0   # borde inferior
        du[i, Ny] = 0.0  # borde superior
    end

    for j in 1:Ny
        du[1, j] = 0.0   # borde izquierdo
        du[Nx, j] = 0.0  # borde derecho
    end
end

# Preparar el problema y parámetros
u0 = copy(u)
p = (Nx, Ny, dx, dy, alpha)
tspan = (0.0, Nt * dt)
prob = ODEProblem(heat_equation!, u0, tspan, p)

# Solución usando Runge-Kutta 4
sol = solve(prob, RK4(), dt=dt)

# Extraer la solución final
u_final = sol[end]

# Crear un gráfico de la distribución final de temperatura
x = range(0, stop=Lx, length=Nx)
y = range(0, stop=Ly, length=Ny)
heatmap(x, y, u_final, title="Distribución final de temperatura", xlabel="x", ylabel="y", colorbar_title="Temperatura")

