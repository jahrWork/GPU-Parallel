using LinearAlgebra, Plots

# Parámetros del problema
nR = 50   # Número de puntos en la dirección radial
ntheta = 50  # Número de puntos en la dirección angular
Rc = 1.0  # Radio del cilindro
Rinf = 4 * Rc  # Radio exterior (donde no hay perturbaciones)
alpha = 0.01  # Difusividad térmica
v = [0.1, 0.1]  # Vector de velocidad en coordenadas cartesianas, se adaptará a polares
dt = 0.001  # Paso de tiempo
tfinal = 1.0  # Tiempo final de la simulación
nsteps = Int(tfinal / dt)  # Número de pasos de tiempo

# Definimos el tamaño de la malla
dR = (Rinf - Rc) / (nR - 1)
dtheta = π / (ntheta - 1)  # Usamos π para representar la mitad del dominio

# Inicialización continua de temperaturas usando una función exponencial
function initialize_temperature(nR, ntheta, dR, dtheta)
    T = zeros(nR, ntheta)
    for i in 1:nR
        for j in 1:ntheta
            R = Rc + (i-1) * dR
            theta = (j-1) * dtheta
            x = R * cos(theta)
            y = R * sin(theta)
            T[i, j] = exp(-(25*(x-0.5)^2 + 25*(y-0.5)^2)) - 0.1
            if T[i, j] < 0
                T[i, j] = 0
            end
        end
    end
    apply_boundary_conditions!(T)
    return T
end

# Función para aplicar las condiciones de frontera (Dirichlet en este caso)
function apply_boundary_conditions!(T)
    T[1, :] .= 100.0  # Condición de temperatura del cilindro
    T[end, :] .= 20.0  # Condición de temperatura en el exterior
    T[:, 1] .= T[:, 2]  # Simetría en θ = 0
    T[:, end] .= T[:, end-1]  # Simetría en θ = π
end

# Definimos los nodos en R y θ
nodes_R = range(Rc, stop=Rinf, length=nR)
nodes_theta = range(0, stop=π, length=ntheta)

# Función para calcular la matriz de diferencias finitas de primera derivada centrada
function finite_diff_matrix_first_derivative(n, d)
    D = zeros(n, n)
    for i in 2:n-1
        D[i, i-1] = -1 / (2 * d)
        D[i, i+1] = 1 / (2 * d)
    end
    return D
end

# Función para calcular la matriz de diferencias finitas de segunda derivada centrada
function finite_diff_matrix_second_derivative(n, d)
    D2 = zeros(n, n)
    for i in 2:n-1
        D2[i, i-1] = 1 / d^2
        D2[i, i] = -2 / d^2
        D2[i, i+1] = 1 / d^2
    end
    return D2
end

# Construcción de las matrices de derivadas en coordenadas polares
D_R = finite_diff_matrix_first_derivative(nR, dR)
D_theta = finite_diff_matrix_first_derivative(ntheta, dtheta)
D2_R = finite_diff_matrix_second_derivative(nR, dR)
D2_theta = finite_diff_matrix_second_derivative(ntheta, dtheta)

# Construcción de las matrices usando productos de Kronecker
I_R = I(nR)
I_theta = I(ntheta)

# Laplaciano en coordenadas polares
Laplacian_R = D2_R + Diagonal(1 ./ nodes_R) * D_R
Laplacian_theta = D2_theta * Diagonal(1 ./ nodes_R.^2)
Laplacian = kron(I_theta, Laplacian_R) + kron(Laplacian_theta, I_R)

# Gradient en coordenadas polares
vr = v[1] * cos.(nodes_theta) + v[2] * sin.(nodes_theta)
vtheta = -v[1] * sin.(nodes_theta) + v[2] * cos.(nodes_theta)
Gradient_R = Diagonal(vr) * D_R
Gradient_theta = Diagonal(1 ./ nodes_R) * D_theta * Diagonal(vtheta)
Gradient = kron(I_theta, Gradient_R) + kron(Gradient_theta, I_R)

# Inicializamos el campo de temperatura
T = initialize_temperature(nR, ntheta, dR, dtheta)
T = vec(T)  # Convertimos T a un vector columna
T_new = similar(T)

# Lista para almacenar los estados de la temperatura
temperaturas = []

# Simulación usando diferencias finitas con convección y difusión
@elapsed begin
    for step in 1:nsteps
        T_new = T .+ dt .* (alpha * (Laplacian * T) - (Gradient * T))
        global T = T_new
        T = reshape(T, nR, ntheta)
        apply_boundary_conditions!(T)
        T = vec(T)
        push!(temperaturas, copy(T))  # Guardamos el estado actual de la temperatura
    end
end

# Convertimos el resultado a una matriz para mostrarlo
T = reshape(T, nR, ntheta)

# Convertir coordenadas polares a cartesianas para la visualización
function polar_to_cartesian(nodes_R, nodes_theta)
    R_mat = repeat(reshape(nodes_R, nR, 1), 1, ntheta)
    theta_mat = repeat(reshape(nodes_theta, 1, ntheta), nR, 1)
    x = R_mat .* cos.(theta_mat)
    y = R_mat .* sin.(theta_mat)
    return x, y
end

# Crear la animación usando Plots
x, y = polar_to_cartesian(nodes_R, nodes_theta)

# Encontrar el mínimo y máximo global de temperatura
z_min = minimum(T)
z_max = maximum(T)

anim = @animate for i in 1:5:length(temperaturas)
    t = reshape(temperaturas[i], nR, ntheta)
    Plots.surface(x, y, t, c=:inferno, clim=(z_min, z_max), title="Distribución de temperatura", xlabel="x", ylabel="y", aspect_ratio=:equal)
end

# Guardar la animación como un gif
gif(anim, "difusion_calor_2d_diferencias_finitas_conveccion.gif", fps=30)
