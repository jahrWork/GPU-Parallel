using Pkg
Pkg.add("Plots")
using LinearAlgebra, SparseArrays, Plots

# Parámetros del problema
nx = 5   # Número de puntos en la dirección x
ny = 5   # Número de puntos en la dirección y
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

# Función para construir los polinomios de Lagrange y sus derivadas
function lagrange_basis(nodes, i, x)
    L = 1.0
    for j in 1:length(nodes)
        if i != j
            L *= (x - nodes[j]) / (nodes[i] - nodes[j])
        end
    end
    return L
end

function lagrange_basis_derivative(nodes, i, x)
    dL = 0.0
    for j in 1:length(nodes)
        if i != j
            term = 1.0 / (nodes[i] - nodes[j])
            for k in 1:length(nodes)
                if k != i && k != j
                    term *= (x - nodes[k]) / (nodes[i] - nodes[k])
                end
            end
            dL += term
        end
    end
    return dL
end

# Función para crear la matriz del operador laplaciano usando polinomios de Lagrange
function create_laplacian_matrix_lagrange(nx, ny, dx, dy)
    N = nx * ny
    L = spzeros(N, N)
    nodes_x = range(0, stop=Lx, length=nx)
    nodes_y = range(0, stop=Ly, length=ny)

    for j in 1:ny
        for i in 1:nx
            index = (j-1)*nx + i
            for jp in 1:ny
                for ip in 1:nx
                    indexp = (jp-1)*nx + ip
                    if i == ip
                        L[index, indexp] += sum(lagrange_basis_derivative(nodes_x, i, nodes_x[k]) for k in 1:nx) / dx^2
                    else
                        L[index, indexp] += lagrange_basis_derivative(nodes_x, i, nodes_x[ip]) / dx^2

                    if j == jp
                        L[index, indexp] += sum(lagrange_basis_derivative(nodes_y, j, nodes_y[k]) for k in 1:ny) / dy^2
                    else
                        L[index, indexp] += lagrange_basis_derivative(nodes_y, j, nodes_y[jp]) / dy^2
                end
            end
        end
    end

    return L
end

# Inicializamos el campo de temperatura
T = initialize_temperature(nx, ny, dx, dy)
T = vec(T)  # Convertimos T a un vector columna
T_new = similar(T)

# Creamos la matriz Laplaciana 2D usando polinomios de Lagrange
L = create_laplacian_matrix_lagrange(nx, ny, dx, dy)

# Lista para almacenar los estados de la temperatura
temperaturas = []

# Ejecutamos la simulación
for step in 1:nsteps
    T_new = T .+ alpha * dt * (L * T)
    T = T_new
    T = reshape(T, nx, ny)
    apply_boundary_conditions!(T)
    T = vec(T)
    push!(temperaturas, copy(T))  # Guardamos el estado actual de la temperatura
end

# Convertimos el resultado a una matriz para mostrarlo
T = reshape(T, nx, ny)

# Determinamos los límites de los ejes
x = range(0, stop=Lx, length=nx)
y = range(0, stop=Ly, length=ny)
z_min, z_max = 0, 1.0  # Límites del eje z (temperatura)

# Creamos la animación 3D
intervalo_animacion = 25
animation = @animate for i in 1:intervalo_animacion:length(temperaturas)
    t = temperaturas[i]
    surface(x, y, reshape(t, nx, ny), title="Distribución de temperatura", xlabel="x", ylabel="y", zlabel="Temperatura", c=:inferno, xlims=(0, Lx), ylims=(0, Ly), zlims=(z_min, z_max))
end

# Guardamos la animación como un gif
gif(animation, "difusion_calor_3d.gif", fps=10)
