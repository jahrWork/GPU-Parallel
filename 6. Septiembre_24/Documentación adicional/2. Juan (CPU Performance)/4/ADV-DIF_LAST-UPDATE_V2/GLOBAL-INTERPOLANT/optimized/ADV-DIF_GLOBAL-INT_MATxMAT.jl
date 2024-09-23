# NOTA: EMULAR UN CILINDRO DE TEMPERATURA CONSTANTE GENERA PROBLEMAS EN 
# LA SUAVIDAD DE LA FORMA DE LA FUNCION SOLUCIÓN; DERIVA A PROBLEMAS
# CUANDO SE USA UN INTERPOLANTE GLOBAL.

using Pkg
Pkg.add("LinearAlgebra")
Pkg.add("SparseArrays")
Pkg.add("Plots")
Pkg.add("Kronecker")
Pkg.add("BenchmarkTools")
using LinearAlgebra, SparseArrays, Plots, Kronecker, BenchmarkTools

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
            if T[i,j] < 0
                T[i,j] = 0
            end
        end
    end
    return T
end

# Función para aplicar las condiciones de frontera y mantener la fuente térmica fija
function apply_boundary_conditions!(T, nx, ny, dx, dy)
    # Mantener la temperatura fija en los bordes
    T[1, :] .= exp(-(25*(-0.5)^2))
    T[end, :] .= exp(-(25*(0.5)^2))
    T[:, 1] .= exp(-(25*(-0.5)^2))
    T[:, end] .= exp(-(25*(0.5)^2))

    # Mantener la fuente térmica fija en el centro con la condición inicial
     for i in 1:nx
         for j in 1:ny
             x = (i-1) * dx
             y = (j-1) * dy
             distance = sqrt((x - 0.5)^2 + (y - 0.5)^2)
             if distance <= 0.05
                 T[i,j] = exp(-(25*(x-0.5)^2 + 25*(y-0.5)^2))
                 #T[i,j] = 1
             end
         end
     end
end



# Definimos los nodos en x e y
nodes_x = range(0, stop=Lx, length=nx)
nodes_y = range(0, stop=Ly, length=ny)

# Cálculo de los polinomios de Lagrange para los x
function lagrange_basis(x_nodes, i, x)
    l_i = 1.0
    for j in 1:length(x_nodes)
        if j != i
            l_i *= (x - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
        end
    end
    return l_i
end

# Función para calcular la derivada del polinomio de Lagrange
function lagrange_derivative(x_nodes, i, x)
    dl_i = 0.0
    for m in 1:length(x_nodes)
        if m != i
            term = 1.0 / (x_nodes[i] - x_nodes[m])
            for j in 1:length(x_nodes)
                if j != i && j != m
                    term *= (x - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
                end
            end
            dl_i += term
        end
    end
    return dl_i
end

# Función para calcular la matriz de derivadas usando polinomios de Lagrange
function lagrange_derivative_matrix(x_nodes)
    n = length(x_nodes)
    D = zeros(n, n)
    for i in 1:n
        for j in 1:n
            D[i, j] = lagrange_derivative(x_nodes, j, x_nodes[i])
        end
    end
    return D
end

# Construcción de las matrices de derivadas con POLINOMIOS DE LAGRANGE
D_x = lagrange_derivative_matrix(nodes_x)
D_y = lagrange_derivative_matrix(nodes_y)

# Las matrices de segunda derivada son simplemente el producto de la matriz de primera derivada con ella misma
D2_x = D_x * D_x
D2_y = D_y * D_y

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
        
        T = T_new
        apply_boundary_conditions!(T, nx, ny, dx, dy)
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

# Creamos la animación 2D con líneas de contorno sin letras
intervalo_animacion = 30
animation = @animate for i in 1:intervalo_animacion:length(temperaturas)
    t = temperaturas[i]
    contourf(x, y, reshape(t, nx, ny), c=:inferno, xlims=(0, Lx), ylims=(0, Ly), zlims=(z_min, z_max), xlabel="", ylabel="", title="", clabels=false)
end

# Guardamos la animación como un gif
gif(animation, "adveccion_difusion_contornos_2d.gif", fps=10)
