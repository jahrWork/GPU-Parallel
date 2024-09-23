using LinearAlgebra, SparseArrays, Plots

function print_matrix(m)
    for row in 1:size(m, 1)
        println(join(m[row, :], " "))
    end
end

#PARÁMETROS DE PROBLEMA
nx = 100   # Número de puntos en la dirección x
ny = 100   # Número de puntos en la dirección y
Lx = 1.0  # Longitud del dominio en la dirección x
Ly = 1.0  # Longitud del dominio en la dirección y
alpha = 0.01  # Difusividad térmica
dt = 0.1  # Paso de tiempo
tfinal = 1  # Tiempo final de la simulación
nsteps = Int(tfinal / dt)  # Número de pasos de tiempo

#TAMAÑO DE MALLA
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)

#CONDICION INICIAL DEL CAMPO DE TEMPERATURA
function initialize_temperature(nx, ny, dx, dy)
    T = zeros(nx, ny)
    for i in 1:nx
        for j in 1:ny
            if 0.4 < i*dx < 0.6 && 0.4 < j*dy < 0.6
                T[i, j] = 1.0
            end
        end
    end
    return T
end

#CONDICIONES DE CONTORNO: TIPO DIRICHLET
function apply_boundary_conditions!(T)
    T[1, :] .= 0.0
    T[end, :] .= 0.0
    T[:, 1] .= 0.0
    T[:, end] .= 0.0
end

#CREACIÓN DE LA MATRIZ LAPLACIANA CON DIFERENCIAS FINITAS CENTRADAS
function create_laplacian_matrix(nx, ny, dx, dy)
    N = nx * ny
    L = spzeros(N, N)
   
    for j in 1:ny
        for i in 1:nx
            index = (j-1)*nx + i
            if i > 1
                L[index, index-1] = 1.0 / dx^2
            end
            if i < nx
                L[index, index+1] = 1.0 / dx^2
            end
            if j > 1
                L[index, index-nx] = 1.0 / dy^2
            end
            if j < ny
                L[index, index+nx] = 1.0 / dy^2
            end
            L[index, index] = -2.0 / dx^2 - 2.0 / dy^2
        end
    end

    return L
end

# Inicializamos el campo de temperatura
T = initialize_temperature(nx, ny, dx, dy)
T = vec(T)  # Convertimos T a un vector columna
T_new = similar(T)

# Creamos la matriz Laplaciana 2D
L = create_laplacian_matrix(nx, ny, dx, dy)

# Lista para almacenar los estados de la temperatura
temperaturas = []

# Ejecutamos la simulación
@elapsed begin

for step in 1:nsteps
    T_new = T .+ alpha * dt * (L * T)
    T = T_new
    T = reshape(T, nx, ny)
    apply_boundary_conditions!(T)
    T = vec(T)
    push!(temperaturas, copy(T))  # Guardamos el estado actual de la temperatura
end

end

#USANDO PKG DE UNICODE PARA MOSTRAR LA MATRIZ L POR LA TERMINAL
print_matrix(L)
# Convertimos el resultado a una matriz para mostrarlo
#T = reshape(T, nx, ny)

# Determinamos los límites de los ejes
#x = range(0, stop=Lx, length=nx)
#y = range(0, stop=Ly, length=ny)
#z_min, z_max = 0, 1.0  # Límites del eje z (temperatura)

# Creamos la animación 3D
#intervalo_animacion = 25
#animation = @animate for i in 1:intervalo_animacion:length(temperaturas)
#    t = temperaturas[i]
#    surface(x, y, reshape(t, nx, ny), title="Distribución de temperatura", xlabel="x", ylabel="y", zlabel="Temperatura", c=:inferno, xlims=(0, Lx), ylims=(0, Ly), zlims=(z_min, z_max))
#end

# Guardamos la animación como un gif
#gif(animation, "difusion_calor_3d.gif", fps=10)
