@testset "basic properties" begin
    Random.seed!(42)
    D,d = 3,2
    stype = DoublePEPSZ2(D)
    A = rand(ComplexF64, D,D,D,D,d)
    M = reshape(ein"abcdi,efghi->aebfcgdh"(A, conj(A)), D^2,D^2,D^2,D^2)

    M_real = real(M)
    M_real = convert_bilayer_Z2(M_real)

    ST = SymmetricType(symmetry=:U1, stype=stype, atype=Array, dtype=Float64)
    MZ2_real = asSymmetryArray(M_real, ST; dir=[-1,1,1,-1])
    M_real′ = asArray(stype, MZ2_real)
    @test M_real ≈ M_real′

    M_imag = imag(M)
    M_imag = convert_bilayer_Z2(M_imag)

    ST = SymmetricType(symmetry=:U1, stype=stype, atype=Array, dtype=Float64)
    MZ2_imag = asSymmetryArray(M_imag, ST; dir=[-1,1,1,-1], q=[1])
    M_imag′ = asArray(stype, MZ2_imag)
    @test M_imag ≈ M_imag′
end

