
module GPUParallel


#push!(LOAD_PATH, "./Physics/")
#push!(LOAD_PATH, "./GUI/")


#using ..lumped_model # ..module_name when module is not installed ???
#using ..graphs


# import Pkg
# Pkg.add( "DifferentialEquations" )
# Pkg.add("Interpolations")



#Pkg.add( path = joinpath(@__DIR__, "/Physics/lumped_model.jl"))
#Pkg.add( path = "./Physics/lumped_model.jl")
#Pkg.add( path = "./GUI/graphs.jl")

#using lumped_model, graphs


include("./Physics/lumped_model.jl")
include("./GUI/graphs.jl")


using .lumped_model 
using .graphs


# Parameter Definition
p = (m = 1.0, k = 4.0, ξ = 0.0)
N = 10

sol = lumped_model.simulation(N, p)

graphs.plot(sol)

graphs.movie(sol, N)



# import Pkg
# Pkg.add( "Plots" ) # It does not work 
using Plots
x = range(0, 10, length=100)
y = sin.(x)
plot(x, y)


end 