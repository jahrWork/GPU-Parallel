module main


#ENV["JULIA_LOAD_PATH"] = "../src/Physics/"
#using .lumped_model
#ENV["JULIA_LOAD_PATH"] = "."

push!(LOAD_PATH, "./Physics/")
push!(LOAD_PATH, "./GUI/")

#using lumped_model: simulation

#include("../src/Physics/lumped_model.jl")
#include("lumped_model.jl")


#import .lumped_model
#using GPUParallel.lumped_model # .module_name when module is not installed 

#import Pkg
#Pkg.add("lumped_model")


using ..lumped_model # ..module_name when module is not installed ???
using ..graphs

#using lumped_model

# Parameter Definition
p = (m = 1.0, k = 4.0, ξ = 0.0)
N = 10

sol = lumped_model.simulation2(N, p)

graphs.plot(sol)

end 