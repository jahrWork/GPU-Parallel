module GPUParallel

include("GPUMulticore/GPUMulticore.jl")

import .GPUMulticore: hello

export hello

end # module
