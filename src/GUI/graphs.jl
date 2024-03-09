module graphs


function plot(U)

  println("Hello from graphs")  
 # println("U=", U)

end 

end 

#= 

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


function draw()

# Plot and make an animation
t = Observable(0.0)
x = @lift sol($t)[1:N]
x_last = @lift sol($t)[N]

#const 
FPS = 40
dt = 1 / FPS
step!(t) = t[] += dt




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


end  =#






#end # module
