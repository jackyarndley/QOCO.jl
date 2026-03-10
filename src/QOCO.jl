module QOCO

export Optimizer

using QOCO_jll
using SparseArrays
using LinearAlgebra

# Global library handle, initialized in __init__
const qoco = Ref{String}()

function __init__()
    qoco[] = QOCO_jll.qoco
    return
end

include("c_api.jl")
include(joinpath("MOI_wrapper", "MOI_wrapper.jl"))

end # module QOCO
