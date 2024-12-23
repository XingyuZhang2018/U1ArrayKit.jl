@testset "general flatten reshape" for ifZ2 in [false]
    # (D,D,D,D,D,D,D,D)->(D^2,D^2,D^2,D^2)
    D, χ = 4, 2
    # a = randinitial(Val(:U1), Array, ComplexF64, D,D,D,D,D,D,D,D; dir = [1,-1,-1,1,-1,1,1,-1])
    indqn = [[-1, 0, 1] for _ in 1:5]
    indims = [[1, 2, 1] for _ in 1:5]
    a = randU1(Array, ComplexF64, D,D,4,D,D; dir=[-1,-1,1,1,1], indqn=indqn, indims=indims, ifZ2=ifZ2)
    a = ein"abcde, fgchi -> gbhdiefa"(a, conj(a))

    # @show size.(a.tensor)[[1,2,3]] a.dims[[1,2,3]]
    indqn = [[-1, 0, 1] for _ in 1:8]
    indims = [[1, 2, 1] for _ in 1:8]
    rea, reinfo = U1reshape(a, D^2,D^2,D^2,D^2; reinfo = (nothing, nothing, nothing, indqn, indims, nothing, nothing))
    rerea = U1reshape(rea, D,D,D,D,D,D,D,D; reinfo = reinfo)[1]
    @test rerea ≈ a

    reinfo2 = U1reshapeinfo((D^2,D^2,D^2,D^2), (D,D,D,D,D,D,D,D), a.dir, indqn, indims, ifZ2)
    rerea2 = U1reshape(rea, D,D,D,D,D,D,D,D; reinfo = reinfo2)[1]
    @test rerea2 ≈ a

    # (χ,D,D,χ) -> (χ,D^2,χ)
    D, χ = 2, 5
    indqn = [[-2, -1, 0, 1, 2], [0, 1], [0, 1], [-2, -1, 0, 1, 2]]
    indims = [[1, 1, 1, 1, 1], [1, 1], [1, 1], [1, 1, 1, 1, 1]]
    a = randU1(Array, ComplexF64, χ,D,D,χ; dir=[-1,1,-1,1], indqn=indqn, indims=indims, ifZ2=ifZ2)
    rea, reinfo  = U1reshape(a, χ,D^2,χ; reinfo = (nothing, nothing, nothing, indqn, indims, nothing, nothing))
    rerea = U1reshape(rea, χ,D,D,χ; reinfo = reinfo)[1]
    @test rerea ≈ a

    reinfo2 = U1reshapeinfo((χ,D^2,χ), (χ,D,D,χ), a.dir, indqn, indims, ifZ2)
    rerea2 = U1reshape(rea, χ,D,D,χ; reinfo = reinfo2)[1]
    @test rerea2 ≈ a
end

@testset "general flatten reshape" for ifZ2 in [false, true]
    # (D,D,D,D,D,D,D,D)->(D^2,D^2,D^2,D^2)
    D = 4
    # a = randinitial(Val(:U1), Array, ComplexF64, D,D,D,D,D,D,D,D; dir = [1,-1,-1,1,-1,1,1,-1])
    indqn = [[0, 1] for _ in 1:5]
    indims = [[1,3],[2,2],[3,1],[3,1],[2,2]]
    a = randU1(Array, ComplexF64, D,D,4,D,D; dir=[-1,-1,1,1,1], indqn=indqn, indims=indims, ifZ2=ifZ2)
    a = ein"abcde, fgchi -> gbhdiefa"(a, conj(a))

    indqn = [[0, 1] for _ in 1:8]
    indims = [indims[2], indims[2], indims[4], indims[4], indims[5], indims[5], indims[1], indims[1]]
    rea, reinfo = U1reshape(a, D^2,D^2,D^2,D^2; reinfo = (nothing, nothing, nothing, indqn, indims, nothing, nothing))
    rerea = U1reshape(rea, D,D,D,D,D,D,D,D; reinfo = reinfo)[1]
    @test rerea ≈ a

    reinfo2 = U1reshapeinfo((D^2,D^2,D^2,D^2), (D,D,D,D,D,D,D,D), a.dir, indqn, indims, ifZ2)
    rerea2 = U1reshape(rea, D,D,D,D,D,D,D,D; reinfo = reinfo2)[1]
    @test rerea2 ≈ a

    # (χ,D,D,χ) -> (χ,D^2,χ)
    D, χ = 3, 5
    indqn = [[0, 1] for _ in 1:4]
    indims = [[2, 3], [1, 2], [2, 1], [2, 3]]
    a = randU1(Array, ComplexF64, χ,D,D,χ; dir=[-1,1,-1,1], indqn=indqn, indims=indims, ifZ2=ifZ2)
    rea, reinfo  = U1reshape(a, χ,D^2,χ; reinfo = (nothing, nothing, nothing, indqn, indims, nothing, nothing))
    rerea = U1reshape(rea, χ,D,D,χ; reinfo = reinfo)[1]
    @test rerea ≈ a

    reinfo2 = U1reshapeinfo((χ,D^2,χ), (χ,D,D,χ), a.dir, indqn, indims, ifZ2)
    rerea2 = U1reshape(rea, χ,D,D,χ; reinfo = reinfo2)[1]
    @test rerea2 ≈ a
end