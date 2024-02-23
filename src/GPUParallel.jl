module GPUParallel

include("GPUMulticore/GPUMulticore.jl")

import .GPUMulticore: hello

export hello

include("Physics/Physics.jl")

import .Physics: one_d_model

export one_d_model

end # module
