using U1ArrayKit
using CUDA
using KrylovKit
using LinearAlgebra
using OMEinsum
using Random
using Test

using U1ArrayKit: AbstractSymmetricArray, blockdiv
using VectorInterface: add!!, inner, scale!!, scale

CUDA.allowscalar(false)

@testset "U1Array.jl" begin
    include("test_sitetype.jl")
    include("test_initial_convert.jl")
    include("test_einsum.jl")
    include("test_vectorinterface.jl")
    include("test_decomposition.jl")
    include("test_u1reshape.jl")
end
