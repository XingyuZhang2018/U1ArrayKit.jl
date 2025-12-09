@testset "basic properties" begin
    Random.seed!(42)
    D,d = 3,2
    stype = DoublePEPSZ2(D)
    A = rand(ComplexF64, D,D,D,D,d)
    M = reshape(ein"abcdi,efghi->aebfcgdh"(A, conj(A)), D^2,D^2,D^2,D^2)

    M_real_orgin = real(M)
    M_real_Z2 = convert_bilayer_Z2(M_real_orgin)
    @test M_real_orgin ≈ invert_bilayer_Z2(M_real_Z2)

    ST = SymmetricType(symmetry=:U1, stype=stype, atype=Array, dtype=Float64)
    MZ2_real = asSymmetryArray(M_real_Z2, ST; dir=[-1,1,1,-1])
    M_real′ = asArray(stype, MZ2_real)
    @test M_real_Z2 ≈ M_real′

    M_imag_orgin = imag(M)
    M_imag_Z2 = convert_bilayer_Z2(M_imag_orgin)
    @test M_imag_orgin ≈ invert_bilayer_Z2(M_imag_Z2)
    @show norm(M_imag_orgin - invert_bilayer_Z2(M_imag_Z2))

    ST = SymmetricType(symmetry=:U1, stype=stype, atype=Array, dtype=Float64)
    MZ2_imag = asSymmetryArray(M_imag_Z2, ST; dir=[-1,1,1,-1], q=[1])
    M_imag′ = asArray(stype, MZ2_imag)
    @test M_imag_Z2 ≈ M_imag′
end