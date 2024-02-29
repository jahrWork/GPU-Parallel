module lumped_model

using DifferentialEquations, GLMakie


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



function simulation() 


# Parameter Definition
p = (m = 1.0, k = 4.0, ξ = 0.0)
N = 10

# Initial Condition
u0 = zeros(2 * N)
u0[N] = 1.5

# Solve the ODE
T = 2π / sqrt(p.k / p.m)
tspan = (0.0, 10T)
prob = ODEProblem(one_d_model, u0, tspan, p)
sol = solve(prob, Tsit5(), reltol = 1e-8, abstol = 1e-8)

# Plot and make an animation
t = Observable(0.0)
x = @lift sol($t)[1:N]
x_last = @lift sol($t)[N]

#const 
FPS = 40
dt = 1 / FPS
step!(t) = t[] += dt

function figure()
    fig = Figure(size = (1000, 200))
    ax = Axis(fig[1, 1])

    x₀ = collect(2:2:(2N))
    y = zeros(length(x₀))

    u = @lift x₀ .+ $x
    u_last = @lift [0.0, x₀[end] + $x_last]

    lines!(ax, u_last, [0.0, 0.0], color = :black, linewidth = 2)
    lines!(ax, [0, 0], [-0.15, 0.15], color = :black, linewidth = 2)
    scatter!(ax, u, y, color = :coral, markersize = 60)

    hidedecorations!(ax)
    hidespines!(ax)

    ylims!(ax, -0.25, 0.25)
    xlims!(ax, -0.05, 2N + 2)

    fig
end

fig = figure()

# Test the animation before recording
for i in 1:(tspan[end] * FPS)
    step!(t)
    sleep(dt)
end

# Record the actual animation
fig = figure()
frames = 1:(tspan[end] * FPS)
dir = (@__DIR__) * "./one_d_lumped_model.gif"

record(fig, dir, frames; framerate = FPS) do i
    step!(t)
end

end 

end 