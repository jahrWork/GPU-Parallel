using LinearAlgebra, SparseArrays, Plots, Kronecker

# Parámetros del problema
R_c = 1.0  # Radio del cilindro
R_inf = 4 * R_c  # Radio exterior (donde no hay perturbaciones)
T_c = 100.0  # Temperatura del cilindro
T_inf = 20.0  # Temperatura del aire
k = 0.6  # Conductividad térmica del cilindro (W/m·K)
h = 10.0  # Coeficiente de transferencia de calor por convección (W/m^2·K)
rho = 7850.0  # Densidad del cilindro (kg/m^3)
cp = 500.0  # Capacidad calorífica del cilindro (J/kg·K)
nR = 50  # Número de puntos en R
ntheta = 50  # Número de puntos en theta
dt = 0.01  # Paso temporal
tfinal = 1.0  # Tiempo final de la simulación
nsteps = Int(tfinal / dt)  # Número de pasos de tiempo

# Función para generar puntos de Chebyshev en el dominio [a, b]
function chebyshev_nodes(a, b, n)
    return 0.5 * ((b - a) * cos.(π * (2 .* (1:n) .- 1) / (2 * n)) .+ (b + a))
end

# Puntos de Chebyshev en R y uniformes en θ
R = chebyshev_nodes(R_c, R_inf, nR)
θ = range(0, stop=π, length=ntheta)

# Ordenamos los puntos R y correspondientemente las filas de T
R_sorted_indices = sortperm(R)
R_sorted = R[R_sorted_indices]

# Inicialización de la matriz de temperaturas
T = zeros(nR, ntheta)
T .= T_inf
T[1, :] .= T_c  # R = Rc

# Función para aplicar las condiciones de frontera (Dirichlet en este caso)
function apply_boundary_conditions!(T)
    T[1, :] .= T_c  # R = Rc
    T[:, 1] .= 0.0  # No flujo en θ = 0
    T[:, end] .= 0.0  # No flujo en θ = π
end

# Calculamos las matrices de segunda derivada
D2_R = lagrange_second_derivative_matrix(R_sorted)
D2_θ = lagrange_second_derivative_matrix(θ)

# Construimos la matriz Laplaciana usando productos de Kronecker
I_R = I(nR)
I_θ = I(ntheta)
L = kron(D2_R, I_θ) + kron(I_R, D2_θ)

# Convertimos T a un vector columna
T = vec(T)
T_new = similar(T)

# Lista para almacenar los estados de la temperatura
temperaturas_vector = []

# Simulación usando diferencias finitas
@elapsed begin
    for step in 1:nsteps
        T_new = T .+ dt * (L * T)
        T = reshape(T_new, nR, ntheta)
        apply_boundary_conditions!(T)
        T = vec(T)
        push!(temperaturas_vector, copy(T))
    end
end

# Convertimos el resultado a una matriz para mostrarlo
T = reshape(T, nR, ntheta)

# Creamos la animación 2D
intervalo_animacion = 5
animation_vector = @animate for i in 1:intervalo_animacion:length(temperaturas_vector)
    t = reshape(temperaturas_vector[i], nR, ntheta)
    heatmap(θ, R_sorted, t', title="Distribución de temperatura", xlabel="θ", ylabel="R", c=:inferno)
end

# Guardamos la animación como un gif
gif(animation_vector, "difusion_calor_2d_chebyshev.gif", fps=5)
