using GPUParallel

hello()

# ALT + J, ALT + O

using GLMakie 
import GLMakie: lines

x = collect(1:0.1:4)
lines(x, x.^2, linewidth = 5)