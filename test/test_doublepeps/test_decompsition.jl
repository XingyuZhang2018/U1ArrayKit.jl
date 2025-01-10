@testset "QR decomposition" begin
    D,d = 9,4
    double = randU1double(Array, D,d)
    Q, R = qrpos(double)
    @test Q * R ≈ double
    @test (Q' * Q).real ≈ IU1double(Array, d).real atol=1e-12

    double = randU1double(Array, D,d,D)
    Q, R = qrpos(reshape(double, D*d,D))
    @test Q * R ≈ double
    @test (Q' * Q).real ≈ IU1double(Array, D).real atol=1e-12

    double = randU1double(Array, d,D)
    L, Q = lqpos(double)
    @test L * Q ≈ double
    @test (Q * Q').real ≈ IU1double(Array, d).real atol=1e-12

    double = randU1double(Array, D,d,D)
    L, Q = lqpos(reshape(double, D,d*D))
    @test L * Q ≈ double
    @test (Q * Q').real ≈ IU1double(Array, D).real atol=1e-12
end