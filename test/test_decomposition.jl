@testset "U1 order-3 tensor qr with $sitetype $atype{$dtype}" for atype in [Array], dtype in [ComplexF64], sitetype in [electronPn(),electronZ2(),tJZ2()]
    Random.seed!(100)
    χ, D = 10, 4
    A = randU1(sitetype, atype, dtype, χ, D, χ; dir = [-1, 1, 1])
	Atensor = asArray(sitetype, A)
	A = reshape(A, χ * D, χ) 
	Atensor = reshape(Atensor, χ * D, χ)
	Q, R = qrpos(A)
    @test Q.qn == sort(Q.qn) 
    @test R.qn == sort(R.qn)
    Qtensor, Rtensor = qrpos(Atensor)
    @test Qtensor * Rtensor ≈ Atensor
	@test Q * R ≈ A

    Q = reshape(Q, χ,D,χ)
    R = reshape(R, χ,χ)
    A = reshape(A, χ,D,χ)
    @test ein"abc,cd->abd"(Q, R) ≈ A

    @test Qtensor' * Qtensor ≈ I(χ)
    M = ein"cda,cdb -> ab"(reshape(Q, χ, D, χ), conj(reshape(Q, χ, D, χ)))
    @test asArray(sitetype, M) ≈ I(χ)

	@test asArray(sitetype, reshape(Q, χ,D,χ)) ≈ reshape(Qtensor, χ,D,χ)
	@test asArray(sitetype, R) ≈ Rtensor
end

@testset "U1 order-N tensor qr with $sitetype $atype{$dtype}" for atype in [Array], dtype in [ComplexF64], sitetype in [electronPn(),electronZ2(),tJZ2()]
    Random.seed!(100)
    χ, D = 4, 4
    A = randU1(sitetype, atype, dtype, χ, D, D, χ; dir = [-1,1,1,1])
	Atensor = asArray(sitetype, A)
	A = reshape(A, χ*D, χ*D) 
	Atensor = reshape(Atensor, χ*D, χ*D)
    
	Q, R = qrpos(A)
    @test Q * R ≈ A

    Q = reshape(Q, χ, D, D*χ)
    R = reshape(R, D*χ, D, χ)
    A = reshape(A, χ, D, D, χ)
	@test ein"abc,cde->abde"(Q, R) ≈ A
end

@testset "U1 lq with $atype{$dtype}" for atype in [Array], dtype in [ComplexF64], sitetype in [electronPn(),electronZ2(),tJZ2()]
    Random.seed!(100)
    χ, D = 10, 4
    A = randU1(sitetype, atype, dtype, χ, D, χ; dir = [-1, 1, 1])
	Atensor = asArray(sitetype, A)
	A = reshape(A, χ, χ*D)
	Atensor = reshape(Atensor, χ, χ*D)
	L, Q = lqpos(A)
    @test Q.qn == sort(Q.qn) 
    @test L.qn == sort(L.qn)
    Ltensor, Qtensor = lqpos(Atensor)
    @test Ltensor*Qtensor ≈ Atensor
	@test L*Q ≈ A

    @test Qtensor*Qtensor' ≈ I(χ)
    M = ein"acd,bcd -> ab"(reshape(Q, χ,D,χ),conj(reshape(Q, χ,D,χ)))
    @test asArray(sitetype, M) ≈ I(χ)

	@test asArray(sitetype, L) ≈ Ltensor
	@test asArray(sitetype, reshape(Q,  χ,D,χ)) ≈ reshape(Qtensor,  χ,D,χ)
end

@testset "U1 order-2 tensor svd with $atype{$dtype} $sitetype" for atype in [Array], dtype in [Float64], sitetype in [electronPn(),electronZ2(),tJZ2()]
    Random.seed!(100)
    χ = 20
    A = randU1(sitetype, atype, dtype, χ, χ; dir = [-1, 1])
	Atensor = asArray(sitetype, A)
	U, S, V = svd!(copy(A))
    Utensor, Stensor, Vtensor = svd!(copy(Atensor))
    @test Utensor * Diagonal(Stensor) * Vtensor' ≈ Atensor
	@test U * Diagonal(S) * V' ≈ A

    U, S, V = svd!(copy(A); trunc=10)
    @test sum(S.dims) == [10, 10]
end

@testset "U1 order-N tensor svd with $atype{$dtype} $sitetype" for atype in [Array], dtype in [Float64], sitetype in [electronPn(),electronZ2(),tJZ2()]
    Random.seed!(100)
    χ,D = 4,4
    A = randU1(sitetype, atype, dtype, χ,D,D,χ; dir = [-1,1,1,1])
    A = reshape(A, χ*D, χ*D)
	U, S, V = svd!(copy(A))
	@test U * Diagonal(S) * V' ≈ A

    U = reshape(U, χ,D,D*χ)
    Vt = reshape(V', χ*D,D,χ)
    A = reshape(A, χ,D,D,χ)
    @test ein"(abc,cd),def->abef"(U, Diagonal(S), Vt) ≈ A
    U, S, V = svd!(copy(A); trunc=5)
    @test sum(S.dims) == [5, 5]
    @test norm(U * Diagonal(S) * V' - A) != 0
end

@testset "invDiagU1Matrix with $sitetype $atype{$dtype}" for atype in [Array], dtype in [ComplexF64], sitetype in [electronPn(),electronZ2(),tJZ2()]
    Random.seed!(100)
    χ = 4
    A = randU1DiagMatrix(sitetype, atype, dtype, χ; dir = [-1, 1])   
    invA = invDiagU1Matrix(A)
    @test A * invA ≈ IU1(sitetype, atype, dtype, χ; dir = [-1, 1])
end