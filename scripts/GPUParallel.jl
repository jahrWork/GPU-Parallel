module GPUParallel


ENV["JULIA_LOAD_PATH"] = "../src/Physics/"

#include("../src/Physics/lumped_model.jl")

using lumped_model
  
end 



lumped_model.simulation()