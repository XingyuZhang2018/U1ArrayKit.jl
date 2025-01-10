@testset "initial_U1_double" begin
    Random.seed!(42)
    D,d = 4,4
    double = randU1double(Array, D,d,D; dir=[-1,1,1])
    @test double isa DoubleArray
    @test size(double) == (D,d,D)

    double = IU1double(Array, D; dir=[-1,1])
    @test double isa DoubleArray
    @test size(double) == (D,D)
    @test asComplexArray(asArray(DoublePEPSZ2(Int(sqrt(D))), double)) == I(D)
end