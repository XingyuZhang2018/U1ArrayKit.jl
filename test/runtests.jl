using U1ArrayKit
using CUDA
using KrylovKit
using LinearAlgebra
using OMEinsum
using Random
using Test
using Zygote

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
    include("test_symmetrictype.jl")
    include("test_autodiff.jl")
    include("test_doublepeps/test_base.jl")
    include("test_doublepeps/test_initial.jl")
    include("test_doublepeps/test_einsum.jl")
    include("test_doublepeps/test_decompsition.jl")
end
