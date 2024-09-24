using Plots
using Printf

# Parameters
nx, ny = 25, 25       # número de puntos en x e y
Lx, Ly = 1.0, 1.0       # dimensiones del dominio
dx, dy = Lx/nx, Ly/ny   # espaciado de la malla
alpha = 0.001        # difusividad térmica
T_inf = 20.0            # temperatura del aire entrante
T_rect = 100.0          # temperatura del rectángulo
u = 0.01            # velocidad del flujo de aire (izquierda a derecha)

# Esabilidad usando CFL 
dt_diff = (dx^2) / (2 * alpha)
dt_conv = dx / u
dt_max = min(dt_diff, dt_conv)
dt = min(dt_max, 0.0001)  # elegir dt estable

println("Paso temporal: $dt")
println("Maximo paso estable: $dt_max")

# Número de pasos
t_end = 1.0             # tiempo final
nsteps = Int(t_end / dt)

# Inicialización del campo de temperatura
T = fill(T_inf, nx+1, ny+1)

# Índices del rectángulo
rect_x = Int(floor(nx/8)):Int(floor(3*nx/8))
rect_y = Int(floor(ny/8)):Int(floor(3*ny/8))
T[rect_x, rect_y] .= T_rect

# Condiciones de contorno
function apply_boundary_conditions!(T, T_inf, u, dt, dx)
    nx, ny = size(T)
    
    # Inflow (left side)
    T[1, :] .= T_inf
    
    # Outflow (right side) - COND CONT Convectiva outflow 
   # for j in 1:ny
   #     T[end, j] = T[end, j] - u * dt / dx * (T[end, j] - T[end-1, j])
    #end
    
    



    # Arriba 
    T[:, end] .= T_inf
    
    # Abajo 
    T[:, 1] .= T_inf
end

# Actualizar el campo de temperatura
function update_temperature!(T, alpha, u, dx, dy, dt)
    nx, ny = size(T)
    T_new = copy(T)
    elaapsedtime= @elapsed begin
        
        for i in 2:nx-1
        for j in 2:ny-1
            # Convección
            convection = -u * (T[i,j] - T[i-1,j]) / dx
 #upwinding
 
 # Tx = Dx T  Incluye todas las derivadas (cualquier orden) ver Matvect/ matmat// para ver comparación(tiempo y resultados) u*gradT
            # Difusión

            # Txx = Dxx T igual evaluar que implementación va más rápido Dxx la calculas una vez 

            diffusion = alpha * (
                (T[i+1,j] - 2*T[i,j] + T[i-1,j]) / dx^2 +
                (T[i,j+1] - 2*T[i,j] + T[i,j-1]) / dy^2 )
            
            T_new[i,j] = T[i,j] + dt * (convection + diffusion)
        end
    end
end
    return T_new
end

# Temperatura del rectángulo
function set_rectangle_temperature!(T, rect_x, rect_y, T_rect)
    T[rect_x, rect_y] .= T_rect
end

# Configuración de la visualización
anim = Animation()

# Bucle de integración en el tiempo
for step in 1:nsteps
    # Aplicar condiciones de contorno
    apply_boundary_conditions!(T, T_inf, u, dt, dx)

    # Actualizar el campo de temperatura
    global T = update_temperature!(T, alpha, u, dx, dy, dt)

    # Fijar la temperatura del rectángulo
    set_rectangle_temperature!(T, rect_x, rect_y, T_rect)

    # Visualización
    if step % 10 == 0
        heatmap(T', c=:inferno, clim=(T_inf, T_rect), title=@sprintf("Time: %.3f s", step*dt))
        frame(anim)
    end
end

# Guardar la animación como GIF
gif(anim, "heat_transfer_simulation1.gif", fps=10)
println("Elapsed time for the simulation loop: ", elapsedtime, " seconds")