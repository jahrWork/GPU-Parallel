using Pkg
Pkg.add("Plots")
using LinearAlgebra, SparseArrays, Plots

function print_matrix(m)
    for row in 1:size(m, 1)
        println(join(m[row, :], " "))
    end
end

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

# Inicialización continua de temperaturas
function initialize_temperature(nx, ny, dx, dy)
    T = zeros(nx, ny)
    for i in 1:nx
        for j in 1:ny
            x = (i-1) * dx
            y = (j-1) * dy
            T[i,j] = exp(-(25*(x-0.5)^2 + 25*(y-0.5)^2))
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


# Función para calcular la matriz de la primera derivada de Lagrange
function lagrange_derivative_matrix(nodes)
    n = length(nodes)
    D = zeros(n, n)
    for i in 1:n
        for j in 1:n
            if i != j
                D[i, j] = 1 / (nodes[i] - nodes[j])
            end
        end
    end
    return D
end

# Función para calcular la matriz de la segunda derivada de Lagrange
function lagrange_second_derivative_matrix(nodes)
    D = lagrange_derivative_matrix(nodes)
    D2 = D * D
    return D2
end

# Construcción de las matrices de segunda derivada en x e y
nodes_x = range(0, stop=Lx, length=nx)
nodes_y = range(0, stop=Ly, length=ny)

D2_x = lagrange_second_derivative_matrix(nodes_x)
D2_y = lagrange_second_derivative_matrix(nodes_y)

# Inicializamos el campo de temperatura
T = initialize_temperature(nx, ny, dx, dy)
T_new = similar(T)

# Lista para almacenar los estados de la temperatura
temperaturas = []

# Ejecutamos la simulación
@elapsed begin
    for step in 1:nsteps

        # d2T_dx2 = D2_x * T
        # d2T_dy2 = T * D2_y'
        # d2T = d2T_dx2 + d2T_dy2

        T_new = T .+ alpha * dt .* (D2_x * T + T * D2_y')
        apply_boundary_conditions!(T_new)
        T = T_new
        push!(temperaturas, copy(T))  # Guardamos el estado actual de la temperatura
    end
end


## === REPRESENTACIÓN GRÁFICA === ##


# Convertimos el resultado a una matriz para mostrarlo
T = reshape(T, nx, ny)

# Determinamos los límites de los ejes
x = range(0, stop=Lx, length=nx)
y = range(0, stop=Ly, length=ny)
z_min, z_max = 0, 1.0  # Límites del eje z (temperatura)

# Creamos la animación 3D
intervalo_animacion = 5
animation = @animate for i in 1:intervalo_animacion:length(temperaturas)
    t = temperaturas[i]
    surface(x, y, t, title="Distribución de temperatura", xlabel="x", ylabel="y", zlabel="Temperatura", c=:inferno, xlims=(0, Lx), ylims=(0, Ly), zlims=(z_min, z_max))
end

# Guardamos la animación como un gif
gif(animation, "difusion_calor_lagrange_matmul_3d.gif", fps=5)
