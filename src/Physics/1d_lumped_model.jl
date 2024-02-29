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
    # F[N] = -k * (x[N] - x[N - 1]) - ξ * (ẋ[N] - ẋ[N - 1]) # Free end
    F[N] = -k * x[N] - k * (x[N] - x[N - 1]) - ξ * ẋ[N] - ξ * (ẋ[N] - ẋ[N - 1]) # Fixed end

    return [ẋ; F ./ m]
end

# x'' + x' + x = 0
# x'' = F(x) / m

# u_1 = x
# u_2 = x'
# x'' = -x' -x = -u_2 - u_1

# d/dt (u_1; u_2) = (u_2; -u_2 - u_1)
