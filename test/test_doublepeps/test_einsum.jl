@testset "DoubleArray einsum" begin
    Random.seed!(42)
    D,d = 3,2
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
