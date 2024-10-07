using Pkg
Pkg.add("GLMakie")
using LinearAlgebra, GLMakie


function heatpropagate(u::AbstractArray{T,N}, timesteps::Integer, αΔtΔx⁻²::Real,
    uprev::AbstractArray{T,N}) where {T<:Number,N}

    αΔtΔx⁻² ≤ 0.5 / N || throw(ArgumentError("αΔtΔx⁻² = $αΔtΔx⁻² violates Fourier stability condition"))

    unitvecs = ntuple(i -> CartesianIndex(ntuple(==(i), Val(N))), Val(N))

    I = CartesianIndices(u)
    Ifirst, Ilast = first(I), last(I)
    I1 = oneunit(Ifirst)

    for t = 1:timesteps

        @inbounds @simd for i in Ifirst+I1:Ilast-I1

            ∇²u = -2N * u[i]
            for uvec in unitvecs
                ∇²u += u[i+uvec] + u[i-uvec]
            end

            u[i] = uprev[i] + αΔtΔx⁻² * ∇²u

        end

        # Here Boundary conditions would be imposed (except if they are Dirichlet at boundary)

        uprev = u
    end

    return u, uprev
end

begin
    L = 1.0
    nx = 25
    α = 0.01
    Fo = 1 / 4

    dx = L / (nx - 1)
    dt = 0.001
    nt = 1000

    x = range(-L / 2, L / 2, length=nx)
    y = range(-L / 2, L / 2, length=nx)
    u = [exp(-((xi^2 + yi^2) / 0.5^2)) for yi in y, xi in x]
    u[1, :] .= 0.0
    u[end, :] .= 0.0
    u[:, 1] .= 0.0
    u[:, end] .= 0.0

    uprev = copy(u)
end

begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="x", ylabel="y")
    hm = heatmap!(ax, x, y, u, colormap=:viridis, colorrange=(0, 1))
    Colorbar(fig[1, 2], hm, label="Temperature")

    fig
end


begin
    u, uprev = heatpropagate(u, nt, Fo, uprev)
    heatmap!(ax, x, y, u, colormap=:viridis, colorrange=(0, 1))
    fig
end

using BenchmarkTools

@benchmark heatpropagate(u, nt, Fo, uprev)
@time heatpropagate(u, nt, Fo, uprev);
