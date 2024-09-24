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
    x = range(-1.0, 1.0, length=100)
    u = @. exp(-(x / 0.2)^2)
    u[1] = u[end] = 0.0

    set_publication_theme!()

    lines(x, u)

    α = 1.0
    Δx = x[2] - x[1]
    Fo = 1 / 2
    Δt = Fo * Δx^2 / α

    uprev = copy(u)
end

# @time heatpropagate(u, 20, Fo, uprev);

for i = 1:50
    u, uprev = heatpropagate(u, 20, Fo, uprev)
    lines!(x, u)
end
