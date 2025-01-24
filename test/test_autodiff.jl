function num_grad(f, K; δ::Real=1e-5)
    if eltype(K) == ComplexF64
        (f(K + δ / 2) - f(K - δ / 2)) / δ + 
            (f(K + δ / 2 * 1.0im) - f(K - δ / 2 * 1.0im)) / δ * 1.0im
    else
        (f(K + δ / 2) - f(K - δ / 2)) / δ
    end
end

function num_grad(f, a::AbstractArray; δ::Real=1e-5)
    b = Array(copy(a))
    df = map(CartesianIndices(b)) do i
        foo = x -> (ac = copy(b); ac[i] = x; f(_arraytype(a)(ac)))
        num_grad(foo, b[i], δ=δ)
    end
    return _arraytype(a)(df)
end

@testset "matrix autodiff with $sitetype $atype{$dtype}" for atype in [Array], dtype in [ComplexF64], sitetype in [electronPn(),electronZ2(),tJZ2()]
    Random.seed!(100)
    A = randU1(sitetype, atype, dtype, 4,4; dir=[-1,1])
    Atensor = asArray(sitetype, A)
    @test asArray(sitetype, Zygote.gradient(norm, A)[1]) ≈ num_grad(norm, Atensor)

    function foo1(x) 
        norm(atype(dtype[x 2x; 3x x]))
    end
    @test Zygote.gradient(foo1, 1)[1] ≈ num_grad(foo1, 1)

    # example to solve differential of array of array
    # use `[]` list then reshape
    A = [randU1(sitetype, atype, dtype, 4,4; dir=[-1,1]) for i in 1:2, j in 1:2]
    function foo2(x)
        # B[i,j] = A[i,j].*x   # mistake
        B = reshape([A[i]*x for i=1:4],2,2)
        return norm(sum(B))
    end
    @test Zygote.gradient(foo2, 1)[1] ≈ num_grad(foo2, 1)
end

@testset "last tr with $sitetype $atype{$dtype}" for atype in [Array], dtype in [ComplexF64], sitetype in [electronPn(),electronZ2(),tJZ2()]
    Random.seed!(100)
    A = randU1(sitetype, atype, dtype, 4,4; dir=[-1,1])
	Atensor = asArray(sitetype, A)
    foo1(x) = norm(tr(x))
    @test asArray(sitetype, Zygote.gradient(foo1, A)[1]) ≈ Zygote.gradient(foo1, Atensor)[1] ≈ num_grad(foo1, Atensor)

    A = randU1(sitetype, atype, dtype, 4,4,4,4; dir = [-1,-1,1,1])
    Atensor = asArray(sitetype, A)
    foo2(x) = norm(ein"abcd,abcd -> "(x,conj(x))[])
    @test asArray(sitetype, Zygote.gradient(foo2, A)[1]) ≈ Zygote.gradient(foo2, Atensor)[1] ≈ num_grad(foo2, Atensor)

    A = randU1(sitetype, atype, dtype, 4,4,4,4; dir = [-1,-1,1,1])
    Atensor = asArray(sitetype, A)
    foo3(x) = norm(ein"abab -> "(x)[])
    foo4(x) = norm(dtr(x))
    @test foo3(Atensor) ≈ foo4(A)
    @test asArray(sitetype, Zygote.gradient(foo4, A)[1]) ≈ Zygote.gradient(foo3, Atensor)[1] ≈ num_grad(foo3, Atensor)
end

@testset "QR factorization with $sitetype $atype{$dtype}" for atype in [Array], dtype in [ComplexF64], sitetype in [electronPn(),electronZ2(),tJZ2()]
    Random.seed!(100)
    M = randU1(sitetype, atype, dtype, 5,3,5; dir = [-1,1,1])
    Mtensor = asArray(sitetype, M)
    function foo(M)
        M = reshape(M, 15, 5)
        Q, R = qrpos(M)
        return norm(Q) + norm(R)
    end
    @test asArray(sitetype, Zygote.gradient(foo, M)[1]) ≈ num_grad(foo, Mtensor)  atol = 1e-8
end

@testset "LQ factorization with $sitetype $atype{$dtype}" for atype in [Array], dtype in [ComplexF64], sitetype in [electronPn(),electronZ2(),tJZ2()]
    Random.seed!(100)
    M = randU1(sitetype, atype, dtype, 5,3,5; dir = [-1,1,1])
    Mtensor = asArray(sitetype, M)
    function foo(M)
        M = reshape(M, 5, 15)
        L, Q = lqpos(M)
        return norm(Q) + norm(L)
    end
    @test asArray(sitetype, Zygote.gradient(foo, M)[1]) ≈ num_grad(foo, Mtensor)  atol = 1e-8
end