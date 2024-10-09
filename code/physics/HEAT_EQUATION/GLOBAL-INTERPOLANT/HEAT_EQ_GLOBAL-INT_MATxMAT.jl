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

function initialize_temperature(nx, ny, dx, dy) #Initial
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
    # Dirichlet conditions
    T[1, :] .= exp(-(25*(-0.5)^2))
    T[end, :] .= exp(-(25*(0.5)^2))
    T[:, 1] .= exp(-(25*(-0.5)^2))
    T[:, end] .= exp(-(25*(0.5)^2))

end

nodes_x = range(0, stop=Lx, length=nx)
nodes_y = range(0, stop=Ly, length=ny)


function lagrange_basis(x_nodes, i, x)
    l_i = 1.0
    for j in 1:length(x_nodes)
        if j != i
            l_i *= (x - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
        end
    end
    return l_i
end


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


D_x = lagrange_derivative_matrix(nodes_x)
D_y = lagrange_derivative_matrix(nodes_y)
D2_x = D_x * D_x
D2_y = D_y * D_y


T = initialize_temperature(nx, ny, dx, dy)  # Convertimos T a un vector columna
temperaturas = []

function simulate_temperature!(T, temperaturas, nsteps, dt, alpha, D2_x, D2_y, nx, ny, dx, dy)
    @inbounds @simd for step in 1:nsteps
        # difusion = alpha * (Laplacian * T)
        # adveccion = -(Gradient * T)
        T = T .+ dt .* (alpha * (D2_x * T + T * D2_y'))
        apply_boundary_conditions!(T, nx, ny, dx, dy)
        push!(temperaturas, T)  # Guardamos el estado actual de la temperatura
    end
end

@elapsed simulate_temperature!(T, temperaturas, nsteps, dt, alpha, D2_x, D2_y, nx, ny, dx, dy)

# Plot
x = range(0, stop=Lx, length=nx);
y = range(0, stop=Ly, length=ny);
z_min, z_max = 0, 1.0;

intervalo_animacion = 30;
animation = @animate for i in 1:intervalo_animacion:length(temperaturas);
    t = temperaturas[i];
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

gif(animation, "adveccion_difusion_contornos_2d.gif", fps=10)
