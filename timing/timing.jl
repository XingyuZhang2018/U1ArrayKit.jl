using BenchmarkTools
using CUDA
using Random
using Test 
using OMEinsum

@testset "OMEinsum with $atype{$dtype} " for atype in [Array, CuArray], dtype in [ComplexF64]
    Random.seed!(100)
    qnd = [0, 1]
    qnD = [0, 1, 2]
    qnχ = [-2, -1, 0, 1, 2]

    dimsd = [1, 1]
    dimsD = [1, 2, 1]
    dimsχ = [1, 4, 6, 4, 1]
    
    d = sum(dimsd)
    D = sum(dimsD)
    χ = sum(dimsχ)

    println("D = $(D) χ = $(χ)")
    AL = randinitial(Val(:U1), atype, dtype, χ, D, D, χ; dir = [-1, -1, 1, 1], indqn = [qnχ, qnD, qnD, qnχ], indims = [dimsχ, dimsD, dimsD, dimsχ], ifZ2 = false)
    M = randinitial(Val(:U1), atype, dtype, D, D, D, D, d; dir = [1,-1,-1,1, 1], indqn = [[qnD for _ in 1:4]..., qnd], indims = [[dimsD for _ in 1:4]..., dimsd], ifZ2 = false)
    FL = randinitial(Val(:U1), atype, dtype, χ, D, D, χ; dir = [1, -1, 1, -1], indqn = [qnχ, qnD, qnD, qnχ], indims = [dimsχ, dimsD, dimsD, dimsχ], ifZ2 = false)

    @time CUDA.@sync FL1 = ein"(((aefi,ijkl),ejgbp),fkhcp),abcd -> dghl"(FL, conj(AL), M, conj(M), AL)
    @time CUDA.@sync FL2 = ein"((aefi,ijkl),(ejgbp,fkhcp)),abcd -> dghl"(FL, conj(AL), M, conj(M), AL)
    @test FL1 ≈ FL2
end

@testset "OMEinsum with $atype{$dtype} " for atype in [Array, CuArray], dtype in [ComplexF64]
    Random.seed!(100)
    qnd = [0, 1]
    qnD = [0, 1]
    qnχ = [0, 1]

    dimsd = [2, 2]
    dimsD = [2, 2]
    dimsχ = [20, 20]
    
    d = sum(dimsd)
    D = sum(dimsD)
    χ = sum(dimsχ)

    println("D = $(D) χ = $(χ)")
    AL = randinitial(Val(:U1), atype, dtype, χ, D, D, χ; dir = [-1, -1, 1, 1], indqn = [qnχ, qnD, qnD, qnχ], indims = [dimsχ, dimsD, dimsD, dimsχ], ifZ2 = true)
    M = randinitial(Val(:U1), atype, dtype, D, D, D, D, d; dir = [1,-1,-1,1, 1], indqn = [[qnD for _ in 1:4]..., qnd], indims = [[dimsD for _ in 1:4]..., dimsd], ifZ2 = true)
    FL = randinitial(Val(:U1), atype, dtype, χ, D, D, χ; dir = [1, -1, 1, -1], indqn = [qnχ, qnD, qnD, qnχ], indims = [dimsχ, dimsD, dimsD, dimsχ], ifZ2 = true)

    @time CUDA.@sync FL1 = ein"(((aefi,ijkl),ejgbp),fkhcp),abcd -> dghl"(FL, conj(AL), M, conj(M), AL)
    @time CUDA.@sync FL2 = ein"((aefi,ijkl),(ejgbp,fkhcp)),abcd -> dghl"(FL, conj(AL), M, conj(M), AL)
    @test FL1 ≈ FL2
end