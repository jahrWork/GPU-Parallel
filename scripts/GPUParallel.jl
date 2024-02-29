module GPUParallel


ENV["JULIA_LOAD_PATH"] = "../src/Physics/"
using .lumped_model

#include("../src/Physics/lumped_model.jl")



lumped_model.simulation()

end 