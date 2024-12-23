@testset "indextoqn and getblockdims" begin
    @test electronPn <: AbstractSiteType
    @test electronSz <: AbstractSiteType
    @test electronZ2 <: AbstractSiteType
    @test tJSz <: AbstractSiteType
    @test tJZ2 <: AbstractSiteType

    @test [indextoqn(electronPn(), i) for i in 1:8] == [0, 1, 1, 2, 1, 2, 2, 3]
    @test getqrange(electronPn(), 8) == [[0,1,2,3]]
    @test getblockdims(electronPn(), 8) == [[1,3,3,1]]

    @test [indextoqn(electronSz(), i) for i in 1:8] == [0, 1, -1, 0, 1, 2, 0, 1]
    @test getqrange(electronSz(), 8) == [[-1,0,1,2]]
    @test getblockdims(electronSz(), 8) == [[1,3,3,1]]

    @test [indextoqn(electronZ2(), i) for i in 1:8] == [0, 1, 1, 0, 1, 0, 0, 1]
    @test getqrange(electronZ2(), 8) == [[0,1]]
    @test getblockdims(electronZ2(), 8) == [[4,4]]

    @test [indextoqn(tJSz(), i) for i in 1:8] == [0, 1, -1, 1, 2, 0, -1, 0]
    @test getqrange(tJSz(), 8) == [[-1,0,1,2]]
    @test getblockdims(tJSz(), 8) == [[2,3,2,1]]

    @test [indextoqn(tJZ2(), i) for i in 1:8] == [0, 1, 1, 1, 0, 0, 1, 0]
    @test getqrange(tJZ2(), 8) == [[0,1]]
    @test getblockdims(tJZ2(), 8) == [[4,4]]
end
