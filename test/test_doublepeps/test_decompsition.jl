@testset "QR decomposition" begin
    D,d = 3,2
    double = randU1double(Array, D,d; dir=[-1,1])
    Q, R = qrpos(double)
    @test Q * R ≈ double
    @test Q' * Q ≈ IU1double(Array, d; dir=[-1,1]) atol=1e-12

    D,d = 3,2
    double = randU1double(Array, D,d,D; dir=[-1,1,1])
    Q, R = qrpos(reshape(double, D^2*d^2,D^2))
    @test Q * R ≈ double
    @test Q' * Q ≈ IU1double(Array, D; dir=[-1,1]) atol=1e-12

    D,d = 3,2
    double = randU1double(Array, d,D; dir=[-1,1])
    L, Q = lqpos(double)
    @test L * Q ≈ double
    @test Q * Q' ≈ IU1double(Array, d; dir=[-1,1]) atol=1e-12

    D,d = 3,2
    double = randU1double(Array, D,d,D; dir=[-1,1,1])
    L, Q = lqpos(reshape(double, D^2,d^2*D^2))
    @test L * Q ≈ double
    @test Q * Q' ≈ IU1double(Array, D; dir=[-1,1]) atol=1e-12
end