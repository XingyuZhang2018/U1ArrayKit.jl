@testset "initial_U1_double" begin
    Random.seed!(42)
    D,d = 2,2
    double = randU1double(Array, D,d,D; dir=[-1,1,1])
    @test double isa DoubleArray
    @test size(double) == (D^2,d^2,D^2)

    double = IU1double(Array, D; dir=[-1,1])
    @test double isa DoubleArray
    @test size(double) == (D^2,D^2)
    @test asComplexArray(asArray(DoublePEPSZ2(D), double)) == I(D^2)
end