using Pkg
Pkg.add("LinearAlgebra")
Pkg.add("SparseArrays")
Pkg.add("Plots")
Pkg.add("Kronecker")
Pkg.add("BenchmarkTools")
using LinearAlgebra, SparseArrays, Plots, Kronecker, BenchmarkTools

nx = 25
ny = 25
Lx = 1.0
Ly = 1.0
alpha = 0.01
v = [0.1, 0.1]
dt = 0.001
tfinal = 1
nsteps = Int(tfinal / dt)

dx = Lx / (nx - 1)
dy = Ly / (ny - 1)

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

function apply_boundary_conditions!(T, nx, ny, dx, dy)

    T[1, :] .= exp(-(25*(-0.5)^2))
    T[end, :] .= exp(-(25*(0.5)^2))
    T[:, 1] .= exp(-(25*(-0.5)^2))
    T[:, end] .= exp(-(25*(0.5)^2))

end

#nodos en x e y
nodes_x = range(0, stop=Lx, length=nx)
nodes_y = range(0, stop=Ly, length=ny)

function finite_diff_matrix_first_derivative(n, dx)
    D = zeros(n, n)
    for i in 2:n-1
        D[i, i-1] = -1 / (2 * dx)
        D[i, i+1] = 1 / (2 * dx)
    end
    return D
end

function finite_diff_matrix_second_derivative(n, dx)
    D2 = zeros(n, n)
    for i in 2:n-1
        D2[i, i-1] = 1 / dx^2
        D2[i, i] = -2 / dx^2
        D2[i, i+1] = 1 / dx^2
    end
    return D2
end

D_x = finite_diff_matrix_first_derivative(nx, dx)
D_y = finite_diff_matrix_first_derivative(ny, dy)
D2_x = finite_diff_matrix_second_derivative(nx, dx)
D2_y = finite_diff_matrix_second_derivative(ny, dy)

T = initialize_temperature(nx, ny, dx, dy)
T_new = similar(T)

# Lista para almacenar los estados de la temperatura
temperaturas = []

# Medir el tiempo del bucle
@elapsed begin
    for step in 1:nsteps

        # difusion = alpha * (Laplacian * T)
        # adveccion = -(Gradient * T)

        T = T .+ dt .* (alpha * (D2_x * T + T * D2_y'))
        apply_boundary_conditions!(T, nx, ny, dx, dy)
        #T = T_new
        push!(temperaturas, T)  # Guardamos el estado actual de la temperatura
    end
end


## === REPRESENTACIÓN GRÁFICA === ##

# Convertimos el resultado a una matriz para mostrarlo
T = reshape(T, nx, ny)

# límites de los ejes
x = range(0, stop=Lx, length=nx)
y = range(0, stop=Ly, length=ny)
z_min, z_max = 0, 1.0  # Límites del eje z (temperatura)

intervalo_animacion = 30
animation = @animate for i in 1:intervalo_animacion:length(temperaturas)
    t = temperaturas[i]
    contourf(x, y, reshape(t, nx, ny);
        c=:inferno,
        xlims=(0, Lx),
        ylims=(0, Ly),
        clim=(z_min, z_max),   
        xlabel="",
        ylabel="",
        title="",
        clabels=false,
        colorbar=true          
    )
end

# Guardamos la animación como un gif
gif(animation, "adveccion_difusion_contornos_2d.gif", fps=10)

function simulate_temperature!(T, temperaturas, nsteps, dt, alpha, D2_x, D2_y, nx, ny, dx, dy)
    @inbounds @simd for step in 1:nsteps
        # Calculamos la nueva temperatura
        T = T .+ dt .* (alpha * (D2_x * T + T * D2_y'))
        
        apply_boundary_conditions!(T, nx, ny, dx, dy)
        push!(temperaturas, T)  # Guardamos el estado actual de la temperatura
    end
end

@benchmark simulate_temperature!(T, temperaturas, nsteps, dt, alpha, D2_x, D2_y, nx, ny, dx, dy)
@time simulate_temperature!(T, temperaturas, nsteps, dt, alpha, D2_x, D2_y, nx, ny, dx, dy);
