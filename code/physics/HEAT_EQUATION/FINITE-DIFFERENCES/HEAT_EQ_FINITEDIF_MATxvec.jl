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

nx = 100
ny = 100
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
    #Dirichlet
    T[1, :] .= exp(-(25*(-0.5)^2))
    T[end, :] .= exp(-(25*(0.5)^2))
    T[:, 1] .= exp(-(25*(-0.5)^2))
    T[:, end] .= exp(-(25*(0.5)^2))

    # Mantener la fuente térmica fija en el centro con la condición inicial
    # for i in 1:nx
    #     for j in 1:ny
    #         x = (i-1) * dx
    #         y = (j-1) * dy
    #         distance = sqrt((x - 0.5)^2 + (y - 0.5)^2)
    #         if distance <= 0.05
    #             #T[i,j] = exp(-(25*(x-0.5)^2 + 25*(y-0.5)^2))
    #             T[i,j] = 1
    #         end
    #     end
    # end
end

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

I_x = I(nx)
I_y = I(ny)
Laplacian = kron(D2_x, I_y) + kron(I_x, D2_y)
Gradient = v[1] * kron(D_x, I_y) + v[2] * kron(I_x, D_y)

T = initialize_temperature(nx, ny, dx, dy)
T = vec(T)
temperaturas = []

function simulate_temperature!(T, temperaturas, nsteps, dt, alpha, Laplacian, Gradient, nx, ny, dx, dy)
    @inbounds @simd for step in 1:nsteps
        #For solving avd-dif equation
        T = T .+ dt .* (alpha * (Laplacian * T))
        #For solving avd-dif equation
        #T = T .+ dt .* (alpha * (Laplacian * T) - (Gradient * T))
        T = reshape(T, nx, ny)
        apply_boundary_conditions!(T, nx, ny, dx, dy)
        T = vec(T)
        push!(temperaturas, copy(T))  # Guardamos el estado actual de la temperatura
    end
end

@elapsed simulate_temperature!(T, temperaturas, nsteps, dt, alpha, Laplacian, Gradient, nx, ny, dx, dy)
# @benchmark simulate_temperature!(T, temperaturas, nsteps, dt, alpha, Laplacian, Gradient, nx, ny, dx, dy)
T = reshape(T, nx, ny)

x = range(0, stop=Lx, length=nx)
y = range(0, stop=Ly, length=ny)
z_min, z_max = 0, 1.0
intervalo_animacion = 30
animation = @animate for i in 1:intervalo_animacion:length(temperaturas)
    t = temperaturas[i]
    contourf(x, y, reshape(t, nx, ny), c=:inferno, xlims=(0, Lx), ylims=(0, Ly), clim=(z_min, z_max), xlabel="", ylabel="", title="", clabels=false, colorbar=true)
end

gif(animation, "adveccion-difusion_finite-diff_MATxvec.gif", fps=10)
