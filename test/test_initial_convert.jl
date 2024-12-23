@testset "U1Array with $sitetype $atype{$dtype}" for atype in [Array], dtype in [ComplexF64], sitetype in [electronPn(),electronZ2(),tJZ2()]
    Random.seed!(100)
    # initial
    @test U1Array <: AbstractSymmetricArray <: AbstractArray

    randinial = randU1(sitetype, atype, dtype, 4,4,5; dir=[-1,1,1])
    @test randinial isa U1Array
    zeroinial = zerosU1(sitetype, atype, dtype, 4,4,5; dir=[-1,1,1])
    Iinial = IU1(sitetype, atype, dtype, 3; dir = [-1,1])
    @test size(randinial) == (4,4,5)
    @test size(zeroinial) == (4,4,5)
    @test size(Iinial) == (3,3)

    # asU1Array and asArray
	A = randU1(sitetype, atype, dtype, 4,4,5; dir=[-1,1,1])
	Atensor = asArray(sitetype, A)
    AA = asU1Array(sitetype, Atensor; dir=[-1,1,1])
    AAtensor = asArray(sitetype, AA)
    @test A ≈ AA
    @test Atensor ≈ AAtensor

	# permutedims
	@test permutedims(Atensor,[3,2,1]) == asArray(sitetype, permutedims(A,[3,2,1]))

	# reshape
	@test reshape(Atensor,(16,5)) == reshape(asArray(sitetype, reshape(reshape(A,16,5),4,4,5)),(16,5))
end

@testset "qndims" begin
    indqn = [[0, 1] for _ in 1:5]
    indims = [[1, 3] for _ in 1:5]
    a = randU1(Array, ComplexF64, sum.(indims)...; dir=[-1,-1,1,1,1], 
                indqn=indqn, indims=indims, ifZ2=true)
    @test all((indqn[i],indims[i]) == qndims(a,i) for i in 1:5)

    indqn = [[0, 1, 2] for _ in 1:5]
    indims = [[1, 3, 2] for _ in 1:5]
    a = randU1(Array, ComplexF64, sum.(indims)...; dir=[-1,-1,1,1,1], 
                indqn=indqn, indims=indims, ifZ2=false)
    
    @test all((indqn[i],indims[i]) == qndims(a,i) for i in 1:5)
end