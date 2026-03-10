using Test

@testset "QOCO" begin
    include("c_wrapper.jl")
    include("MOI_wrapper.jl")
end
