module GPUParallel

#export hello
#include("../src/Physics/Physics.jl")

#include("../src/Physics/lumped_model.jl")

#include("/path/to/module/MyModule.jl")

#ENV["JULIA_LOAD_PATH"] = "../src/Physics/"

include("../src/Physics/lumped_model.jl")
#import lumped_model: lumped_model_sim

#import .Physics: one_d_model

#export one_d_model

end # module


#using lumped_model



lumped_model.simulation()