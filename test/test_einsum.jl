@testset "OMEinsum U1 with $sitetype $atype{$dtype}" for atype in [Array], dtype in [ComplexF64], sitetype in [electronPn(),electronZ2(),tJZ2()]
    Random.seed!(100)
    A = randU1(sitetype, atype, dtype, 3,3,4; dir=[1,1,-1])
    B = randU1(sitetype, atype, dtype, 4,3; dir=[1,-1])
    Atensor = asArray(sitetype, A)
    Btensor = asArray(sitetype, B)

    # binary contraction
    @test ein"abc,cd -> abd"(Atensor,Btensor) ≈ asArray(sitetype, ein"abc,cd -> abd"(A,B))
    @test ein"abc,db -> adc"(Atensor,Btensor) ≈ asArray(sitetype, ein"abc,db -> adc"(A,B))
    @test ein"cba,dc -> abd"(Atensor,Btensor) ≈ asArray(sitetype, ein"cba,dc -> abd"(A,B))
    @test ein"abc,cb -> a"(Atensor,Btensor) ≈ asArray(sitetype, ein"abc,cb -> a"(A,B))
    @test ein"bac,cb -> a"(Atensor,Btensor) ≈ asArray(sitetype, ein"bac,cb -> a"(A,B))
    @test ein"cba,ab -> c"(Atensor,Btensor) ≈ asArray(sitetype, ein"cba,ab -> c"(A,B))
    a = randU1(sitetype, atype, dtype, 3,7,5; dir=[1,-1,1])
    b = randU1(sitetype, atype, dtype, 7,5,3; dir=[1,-1,-1])
    c = ein"abc,bcd->ad"(a,b)
    # @show a b c
    atensor = asArray(sitetype, a)
    btensor = asArray(sitetype, b)
    ctensor = asArray(sitetype, c)
    @test ctensor ≈ ein"abc,bcd->ad"(atensor,btensor)

    # NestedEinsum
    C = randU1(sitetype, atype, dtype, 4,3; dir = [-1,1])
    Ctensor = asArray(sitetype, C)
    @test ein"(abc,cd),ed -> abe"(Atensor,Btensor,Ctensor) ≈ asArray(sitetype, ein"abd,ed -> abe"(ein"abc,cd -> abd"(A,B),C)) ≈ asArray(sitetype, ein"(abc,cd),ed -> abe"(A,B,C))

    # constant
    D = randU1(sitetype, atype, dtype, 3,3,4; dir = [-1,-1,1])
    Dtensor = asArray(sitetype, D)
    @test Array(ein"abc,abc ->"(Atensor,Dtensor))[] ≈ Array(ein"abc,abc ->"(A,D))[]

    # tr
    B = randU1(sitetype, atype, dtype, 4,4; dir = [1,-1])
    Btensor = asArray(sitetype, B)
    @test Array(ein"aa ->"(Btensor))[] ≈ Array(ein"aa ->"(B))[] 
    B = randU1(sitetype, atype, dtype, 4,4,4,4; dir = [-1,-1,1,1])
    Btensor = asArray(sitetype, B)
    @test Array(ein"abab -> "(Btensor))[] ≈ dtr(B)  

    # TeneT unit
    d = 4
    D = 10
    AL = randU1(sitetype, atype, dtype, D,d,D; dir = [-1,1,1])
    M = randU1(sitetype, atype, dtype, d,d,d,d; dir = [-1,1,1,-1])
    FL = randU1(sitetype, atype, dtype, D,d,D; dir = [1,1,-1])
    tAL, tM, tFL = map(x->asArray(sitetype, x), [AL, M, FL])
    tFL = ein"((adf,abc),dgeb),fgh -> ceh"(tFL,tAL,tM,conj(tAL))
    FL = ein"((adf,abc),dgeb),fgh -> ceh"(FL,AL,M,conj(AL))
    @test tFL ≈ asArray(sitetype, FL)
         
    # autodiff test
    D,d = 4,3
    FL = randU1(sitetype, atype, dtype, D, d, D; dir = [1,1,1])
    S = randU1(sitetype, atype, dtype, D, d, D, D, d, D; dir = [-1,-1,-1,-1,-1,-1])
    FLtensor = asArray(sitetype, FL)
    Stensor = asArray(sitetype, S)
    @test ein"(abc,abcdef),def ->"(FL, S, FL)[] ≈ ein"(abc,abcdef),def ->"(FLtensor, Stensor, FLtensor)[]
end