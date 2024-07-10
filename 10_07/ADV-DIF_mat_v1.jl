using Pkg
Pkg.add("Plots")
Pkg.add("Kronecker")
using LinearAlgebra, SparseArrays, Plots, Kronecker, BenchmarkTools

function print_matrix(m)
    for row in 1:size(m, 1)
        println(join(m[row, :], " "))
    end
end

# Parámetros del problema
nx = 100   # Número de puntos en la dirección x
ny = 100   # Número de puntos en la dirección y
Lx = 1.0  # Longitud del dominio en la dirección x
Ly = 1.0  # Longitud del dominio en la dirección y
alpha = 0.01  # Difusividad térmica
v = [0.1, 0.1]  # Vector de velocidad (vx, vy)
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

# Definimos los nodos en x e y
nodes_x = range(0, stop=Lx, length=nx)
nodes_y = range(0, stop=Ly, length=ny)

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
    return D * D
end

# Calculamos las matrices de primera y segunda derivada
D_x = lagrange_derivative_matrix(nodes_x)
D_y = lagrange_derivative_matrix(nodes_y)
D2_x = lagrange_second_derivative_matrix(nodes_x)
D2_y = lagrange_second_derivative_matrix(nodes_y)

# Construcción de las matrices usando productos de Kronecker
        # I_x = I(nx)
        # I_y = I(ny)
        # Laplacian = kron(D2_x, I_y) + kron(I_x, D2_y)
        # Gradient = v[1] * kron(D_x, I_y) + v[2] * kron(I_x, D_y)

# Inicializamos el campo de temperatura
T = initialize_temperature(nx, ny, dx, dy)  # Convertimos T a un vector columna
T_new = similar(T)

# Lista para almacenar los estados de la temperatura
temperaturas = []

# Medir el tiempo del bucle
@elapsed begin
    for step in 1:nsteps

        # difusion = alpha * (Laplacian * T)
        # adveccion = -(Gradient * T)

        T_new = T .+ dt .* (alpha * (D2_x * T + T * D2_y') - (v[1] * (D_x * T) + v[2] * (T * D_y')))
        apply_boundary_conditions!(T)
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
    surface(x, y, reshape(t, nx, ny), title="Distribución de temperatura", xlabel="x", ylabel="y", zlabel="Temperatura", c=:inferno, xlims=(0, Lx), ylims=(0, Ly), zlims=(z_min, z_max))
end

# Guardamos la animación como un gif
gif(animation, "adveccion_difusion_lagrange_mat_3d.gif", fps=5)
