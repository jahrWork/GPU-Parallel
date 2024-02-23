using DifferentialEquations, GLMakie, GPUParallel

# Parameter Definition
p = (m = 0.1, k = 50.0, ξ = 0.5)
N = 100

# Initial Condition
u0 = zeros(2 * N)
u0[(N ÷ 2 - (N ÷ 10)):(N ÷ 2 + (N ÷ 10))] .= 1.0

# Solve the ODE
T = 2π / sqrt(p.k / p.m)
tspan = (0.0, 100T)
prob = ODEProblem(one_d_model, u0, tspan, p)
sol = solve(prob, Tsit5(), reltol = 1e-10, abstol = 1e-10)

# Plot and make an animation
t = Observable(0.0)
y = @lift sol($t)[1:N]
# x_last = @lift sol($t)[N]

const FPS = 40
dt = 1 / FPS
step!(t) = t[] += dt

function figure()
    fig = Figure(size = (1000, 200))
    ax = Axis(fig[1, 1])

    x = 2:2:(2N)

    # u = @lift x₀ .+ $x
    # u_last = @lift [0.0, x₀[end] + $x_last]

    # lines!(ax, u_last, [0.0, 0.0], color = :black, linewidth = 2)
    lines!(ax, [0, 0], [-1, 1], color = :black, linewidth = 2)
    scatter!(ax, x, y, color = :coral, markersize = 10)

    hidedecorations!(ax)
    hidespines!(ax)

    ylims!(ax, -1.05, 1.05)
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
dir = (@__DIR__) * "/videos/square_wave.gif"

record(fig, dir, frames; framerate = FPS) do i
    step!(t)
end
