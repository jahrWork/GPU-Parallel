
using LinearAlgebra, Plots

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
v = [0.1, 0.0]  # Vector de velocidad (vx, vy)
dt = 0.001  # Paso de tiempo
tfinal = 1  # Tiempo final de la simulación
nsteps = Int(tfinal / dt)  # Número de pasos de tiempo

# Definimos el tamaño de la malla
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)

# Inicialización de temperaturas con un círculo de temperatura constante en el centro
function initialize_temperature(nx, ny, dx, dy)
    T = zeros(nx, ny)
    for i in 1:nx
        for j in 1:ny
            x = (i-1) * dx
            y = (j-1) * dy
            if (x-0.5)^2 + (y-0.5)^2 <= 0.1^2
                T[i,j] = 1.0
            else
                T[i,j] = 0.0
            end
        end
    end
    apply_boundary_conditions!(T)
    return T
end

# Función para aplicar las condiciones de frontera (Dirichlet en este caso)
function apply_boundary_conditions!(T)
    T[1, :] .= 0.0
    T[end, :] .= 0.0
    T[:, 1] .= 0.0
    T[:, end] .= 0.0
end

# Función para aplicar la condición de temperatura constante en el círculo central
function apply_constant_circle_temperature!(T, nx, ny, dx, dy)
    for i in 1:nx
        for j in 1:ny
            x = (i-1) * dx
            y = (j-1) * dy
            if (x-0.5)^2 + (y-0.5)^2 <= 0.1^2
                T[i,j] = 1.0
            end
        end
    end
end

# Definimos los nodos en x e y
nodes_x = range(0, stop=Lx, length=nx)
nodes_y = range(0, stop=Ly, length=ny)

# Función para calcular la matriz de diferencias finitas de primera derivada centrada
function finite_diff_matrix_first_derivative(n, dx)
    D = zeros(n, n)
    for i in 2:n-1
        D[i, i-1] = -1 / (2 * dx)
        D[i, i+1] = 1 / (2 * dx)
    end
    return D
end

# Función para calcular la matriz de diferencias finitas de segunda derivada centrada
function finite_diff_matrix_second_derivative(n, dx)
    D2 = zeros(n, n)
    for i in 2:n-1
        D2[i, i-1] = 1 / dx^2
        D2[i, i] = -2 / dx^2
        D2[i, i+1] = 1 / dx^2
    end
    return D2
end

# Construcción de las matrices de derivadas
D_x = finite_diff_matrix_first_derivative(nx, dx)
D_y = finite_diff_matrix_first_derivative(ny, dy)
D2_x = finite_diff_matrix_second_derivative(nx, dx)
D2_y = finite_diff_matrix_second_derivative(ny, dy)

# Construcción de las matrices usando productos de Kronecker
I_x = I(nx)
I_y = I(ny)
Laplacian = kron(D2_x, I_y) + kron(I_x, D2_y)
Gradient = v[1] * kron(D_x, I_y) + v[2] * kron(I_x, D_y)

# Inicializamos el campo de temperatura
T = initialize_temperature(nx, ny, dx, dy)
T = vec(T)  # Convertimos T a un vector columna
T_new = similar(T)

# Lista para almacenar los estados de la temperatura
temperaturas = []

# Medir el tiempo del bucle
@elapsed begin
    for step in 1:nsteps

        T_new = T .+ dt .* (alpha * (Laplacian * T) - (Gradient * T))
        T = T_new
        T = reshape(T, nx, ny)
        apply_boundary_conditions!(T)
        apply_constant_circle_temperature!(T, nx, ny, dx, dy)  # Aplicamos la temperatura constante en el círculo
        T = vec(T)
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
intervalo_animacion = 10
animation = @animate for i in 1:intervalo_animacion:length(temperaturas)
    t = temperaturas[i]
    surface(x, y, reshape(t, nx, ny), title="Distribución de temperatura [M][v]", xlabel="x", ylabel="y", zlabel="Temperatura", c=:inferno, xlims=(0, Lx), ylims=(0, Ly), zlims=(z_min, z_max))
end

# Guardamos la animación como un gif
gif(animation, "adveccion_difusion_diferencias_finitas_matxvec_3d.gif", fps=30)