function one_d_model(u, p, t)
    m, k, ξ = p

    N = length(u) ÷ 2
    F = zeros(N)

    @views x, ẋ = u[1:N], u[(N + 1):end]

    # Equations of motions of the masses
    @views for i in 2:(N - 1)
        F[i] = -k * (x[i] - x[i + 1]) - k * (x[i] - x[i - 1]) - ξ * (ẋ[i] - ẋ[i + 1]) -
               ξ * (ẋ[i] - ẋ[i - 1])
    end

    # First mass
    F[1] = -k * x[1] - k * (x[1] - x[2]) - ξ * ẋ[1] - ξ * (ẋ[1] - ẋ[2])

    # Last mass
    F[N] = -k * (x[N] - x[N - 1]) - ξ * (ẋ[N] - ẋ[N - 1])

    return [ẋ; F ./ m]
end
