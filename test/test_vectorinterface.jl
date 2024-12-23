@testset "vectorinterface function with $sitetype $atype{$dtype}" for atype in [Array], dtype in [ComplexF64], sitetype in [electronPn(),electronZ2(),tJZ2()]
    Random.seed!(100) 
    d = 2
    χ = 5

    A = randU1(sitetype, atype, dtype, χ,d,χ; dir = [-1,1,1])  
    B = randU1(sitetype, atype, dtype, χ,d,χ; dir = [-1,1,1])
    @test inner(A, B) ≈ inner(asArray(sitetype, A), asArray(sitetype, B))

    Acopy = copy(A)
    @test A*3.0 + B*2.0 == add!!(A, B, 2.0, 3.0)
    @test Acopy.tensor != A.tensor

    Acopy = copy(A)
    @test A*2.0 == scale!!(A, 2.0)
    @test Acopy.tensor != A.tensor

    Acopy = copy(A)
    @test B*2.0 == scale!!(A, B, 2.0)
    @test Acopy.tensor != A.tensor

    Acopy = copy(A)
    @test A*2.0 == scale(A, 2.0)
    @test Acopy.tensor == A.tensor
end

@testset "KrylovKit with $atype{$dtype}" for atype in [Array], dtype in [ComplexF64], sitetype in [electronPn(),electronZ2(),tJZ2()]
    Random.seed!(100)
    χ, d = 5, 3
    AL = randU1(sitetype, atype, dtype, χ,d,χ; dir = [-1,1,1])
    M = randU1(sitetype, atype, dtype, d,d,d,d; dir = [-1,1,1,-1])
    FL = randU1(sitetype, atype, dtype, χ,d,χ; dir = [1,1,-1])
    tAL = asArray(sitetype, AL)
    tM = asArray(sitetype, M)
    tFL = asArray(sitetype, FL)

    λs, FLs, info = eigsolve(FL -> ein"((adf,abc),dgeb),fgh -> ceh"(FL,AL,M,conj(AL)), FL, 1, :LM; ishermitian = false)
    tλs, tFLs, info = eigsolve(tFL -> ein"((adf,abc),dgeb),fgh -> ceh"(tFL,tAL,tM,conj(tAL)), tFL, 1, :LM; ishermitian = false)
    @test λs[1] ≈ tλs[1]
    @test asArray(sitetype, FLs[1]) ≈ tFLs[1] 

    λl,FL = λs[1], FLs[1]
    dFL = randU1(sitetype, atype, dtype, χ,d,χ; dir = [1,1,-1])
    dFL -= Array(ein"abc,abc ->"(conj(FL), dFL))[] * FL
    ξl, info = linsolve(FR -> ein"((ceh,abc),dgeb),fgh -> adf"(FR, AL, M, conj(AL)), conj(dFL), -λl, 1) 
    tλl, tFL = tλs[1], tFLs[1]
    tdFL = asArray(sitetype, dFL)
    tξl, info = linsolve(tFR -> ein"((ceh,abc),dgeb),fgh -> adf"(tFR, tAL, tM, conj(tAL)), conj(tdFL), -tλl, 1)
    @test asArray(sitetype, ξl) ≈ tξl
end