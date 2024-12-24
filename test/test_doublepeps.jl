@testset "basic properties" begin
    Random.seed!(42)
    D,d = 2,1
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

@testset "DoubleArray einsum" begin
    Random.seed!(42)
    D,d = 2,1
    stype = DoublePEPSZ2(D)
    A = rand(ComplexF64, D,D,D,D,d)
    M = reshape(ein"abcdi,efghi->aebfcgdh"(A, conj(A)), D^2,D^2,D^2,D^2)

    M_double = DoubleArray(M)
    @test asComplexArray(M_double) ≈ M 
    @test asComplexArray(ein"abcd,cdef->abef"(M_double, M_double)) ≈ ein"abcd,cdef->abef"(M, M)

    M_double_Z2 = convert_bilayer_Z2(M_double)
    ST = SymmetricType(symmetry=:U1, stype=stype, atype=Array, dtype=Float64)
    M_double_Z2_U1Array = asSymmetryArray(M_double_Z2, ST; dir=[-1,1,1,-1])

    # simple einsum
    M_double_Z2_U1Array_einsum = ein"abcd,cdef->abef"(M_double_Z2_U1Array, M_double_Z2_U1Array)
    @test asComplexArray(asArray(stype, M_double_Z2_U1Array_einsum)) ≈ asComplexArray(ein"abcd,cdef->abef"(M_double_Z2, M_double_Z2))

    # permuted einsum
    M_double_Z2_U1Array_einsum = ein"abcd,dcfe->abef"(M_double_Z2_U1Array, M_double_Z2_U1Array)
    @test asComplexArray(asArray(stype, M_double_Z2_U1Array_einsum)) ≈ asComplexArray(ein"abcd,dcfe->abef"(M_double_Z2, M_double_Z2))

    # nested einsum
    M_double_Z2_U1Array_einsum = ein"(abcd,cdef),efgh->abgh"(M_double_Z2_U1Array, M_double_Z2_U1Array, M_double_Z2_U1Array)
    @test asComplexArray(asArray(stype, M_double_Z2_U1Array_einsum)) ≈ asComplexArray(ein"(abcd,cdef),efgh->abgh"(M_double_Z2, M_double_Z2, M_double_Z2))
end