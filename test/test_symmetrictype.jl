# test SymmetricType
@testset "initial with $symmetry $stype $atype{$dtype}" for symmetry in [:U1, :none], atype in [Array], dtype in [ComplexF64], stype in [electronPn(),electronZ2(),tJZ2()]
    ST = SymmetricType(symmetry=symmetry, stype=stype, atype=atype, dtype=dtype)
    if symmetry == :U1
        @test randinitial(ST, 2, 2; dir = [-1,1]) isa U1Array
        @test Iinitial(ST, 2; dir = [-1,1]) isa U1Array
        @test zerosinitial(ST, 2, 2; dir = [-1,1]) isa U1Array
    else
        @test randinitial(ST, 2, 2) isa atype
        @test Iinitial(ST, 2) isa atype
        @test zerosinitial(ST, 2, 2) isa atype
    end
end

@testset "asSymmetryArray with $symmetry $stype $atype{$dtype}" for symmetry in [:U1, :none], atype in [Array], dtype in [ComplexF64], stype in [electronPn(),electronZ2(),tJZ2()]
    ST = SymmetricType(symmetry=symmetry, stype=stype, atype=atype, dtype=dtype)
    A = rand(dtype, 2, 2)
    if symmetry == :U1
        @test asSymmetryArray(A, ST; dir=[-1,1]) isa U1Array
    else
        @test asSymmetryArray(A, ST) isa atype
    end
end