abstract type AbstractSymmetricArray{T,N} <: AbstractArray{T,N} end

"""
    U1Array{T, N}

a struct to hold the N-order U1 tensors
- `qn`(`quantum number`): `N`-length Array
- `dir`(`out or in`): +1 or -1
- `tensor`: continuous storage block tensor ordered by qn
- `size`  : original none-symetric array size
- `dims` size of `tensor`
- `division`: division location for reshape
- `ifZ2`: whether Z2 symmetry
"""
struct U1Array{T, N} <: AbstractSymmetricArray{T,N}
    qn::Vector{Vector{Int}}
    dir::Vector{Int}
    tensor::AbstractArray{T}
    size::Tuple{Vararg{Int, N}}
    dims::Vector{Vector{Int}}
    division::Int
    ifZ2::Bool
    function U1Array(qn::Vector{Vector{Int}}, dir::Vector{Int}, tensor::AbstractArray{T}, size::Tuple{Vararg{Int, N}}, dims::Vector{Vector{Int}}, division::Int, ifZ2::Bool) where {T,N}
        # @show qn sort(unique(collect(Iterators.flatten(qn)))) map(x->sum(x) % 2==0, qn)
        ifZ2 && all(map(x->sum(x) % 2==0, qn))
        new{T, N}(qn, dir, tensor, size, dims, division, ifZ2)
    end
end

size(A::U1Array) = A.size
size(A::U1Array, a) = size(A)[a]
getdir(A::U1Array) = A.dir
conj(A::U1Array) = U1Array(A.qn, -A.dir, conj(A.tensor), A.size, A.dims, A.division, A.ifZ2)
# map(conj, A::U1Array) = conj(A)
conj!(A::U1Array) = U1Array(A.qn, -A.dir, conj!(A.tensor), A.size, A.dims, A.division, A.ifZ2)
norm(A::U1Array) = norm(A.tensor)
norm(A::U1Array, p::Real) = norm(A.tensor, p)
normalize!(A::U1Array) = (normalize!(A.tensor); A)
normalize!(A::U1Array, p::Real) = (normalize!(A.tensor, p); A)
copy(A::U1Array) = U1Array(map(copy, A.qn), copy(A.dir), copy(A.tensor), A.size, A.dims, A.division, A.ifZ2)
similar(A::U1Array) = U1Array(map(copy, A.qn), copy(A.dir), similar(A.tensor), A.size, A.dims, A.division, A.ifZ2)
similar(A::U1Array, atype) = U1Array(map(copy, A.qn), copy(A.dir), atype(similar(A.tensor)), A.size, A.dims, A.division, A.ifZ2)

*(A::U1Array, B::Number) = U1Array(A.qn, A.dir, A.tensor * B, A.size, A.dims, A.division, A.ifZ2)
*(B::Number, A::U1Array) = A * B
rmul!(A::U1Array, B::Number) = (A.tensor .*= B; A)
/(A::U1Array, B::Number) = U1Array(A.qn, A.dir, A.tensor / B, A.size, A.dims, A.division, A.ifZ2)
/(A::Number, B::U1Array) = U1Array(B.qn, B.dir, A / B.tensor, B.size, B.dims, B.division, B.ifZ2)

# broadcasted(*, A::U1Array, B::Number) = U1Array(A.qn, A.dir, A.tensor .* B, A.size, A.dims, A.division, A.ifZ2)
# broadcasted(*, B::Number, A::U1Array) = U1Array(A.qn, A.dir, A.tensor .* B, A.size, A.dims, A.division, A.ifZ2)
# broadcasted(/, A::U1Array, B::Number) = A / B
# broadcasted(/, A::Number, B::U1Array) = U1Array(B.qn, B.dir, A ./ B.tensor, B.size, B.dims, B.division, B.ifZ2)

function +(A::U1Array, B::U1Array)
    if B.qn == A.qn
        U1Array(A.qn, A.dir, A.tensor + B.tensor, A.size, A.dims, A.division, A.ifZ2)
    else
        Aqn, Adims, Atensor = A.qn, A.dims, A.tensor
        Bqn, Bdims, Btensor = B.qn, B.dims, B.tensor
        Abdiv = blockdiv(Adims)
        Bbdiv = blockdiv(Bdims)
        qn = intersect(Aqn, Bqn)
        tensor = Atensor[vcat(Abdiv[indexin(qn, Aqn)]...)] + Btensor[vcat(Bbdiv[indexin(qn, Bqn)]...)]
        dims = A.dims[indexin(qn, Aqn)]
        extraqn = setdiff(Aqn, Bqn)            # setdiff result is dependent on order
        if length(extraqn) !== 0
            push!(qn, extraqn...)
            extraind = indexin(extraqn, Aqn) 
            push!(dims, Adims[extraind]...)
            tensor = [tensor; Atensor[vcat(Abdiv[extraind]...)]]
        end
        extraqn = setdiff(Bqn, Aqn)
        if length(extraqn) !== 0
            push!(qn, extraqn...)
            extraind = indexin(extraqn, Bqn)
            push!(dims, Bdims[extraind]...)
            tensor = [tensor; Btensor[vcat(Bbdiv[extraind]...)]]
        end

        p = sortperm(qn)
        bdiv = blockdiv(dims)
        tensor = tensor[vcat(bdiv[p]...)]
        U1Array(qn[p], A.dir, tensor, A.size, dims[p], A.division, A.ifZ2)
    end
end

function -(A::U1Array, B::U1Array)
    if B.qn == A.qn
        U1Array(A.qn, A.dir, A.tensor - B.tensor, A.size, A.dims, A.division, A.ifZ2)
    else
        Aqn, Adims, Atensor = A.qn, A.dims, A.tensor
        Bqn, Bdims, Btensor = B.qn, B.dims, B.tensor
        Abdiv = blockdiv(Adims)
        Bbdiv = blockdiv(Bdims)
        qn = intersect(Aqn, Bqn)
        tensor = Atensor[vcat(Abdiv[indexin(qn, Aqn)]...)] - Btensor[vcat(Bbdiv[indexin(qn, Bqn)]...)]
        dims = A.dims[indexin(qn, Aqn)]
        extraqn = setdiff(Aqn, Bqn)            # setdiff result is dependent on order
        if length(extraqn) !== 0
            push!(qn, extraqn...)
            extraind = indexin(extraqn, Aqn) 
            push!(dims, Adims[extraind]...)
            tensor = [tensor; Atensor[vcat(Abdiv[extraind]...)]]
        end
        extraqn = setdiff(Bqn, Aqn)
        if length(extraqn) !== 0
            push!(qn, extraqn...)
            extraind = indexin(extraqn, Bqn)
            push!(dims, Bdims[extraind]...)
            tensor = [tensor; -Btensor[vcat(Bbdiv[extraind]...)]]
        end

        p = sortperm(qn)
        bdiv = blockdiv(dims)
        tensor = tensor[vcat(bdiv[p]...)]
        U1Array(qn[p], A.dir, tensor, A.size, dims[p], A.division, A.ifZ2)
    end
end

-(A::U1Array) = U1Array(A.qn, A.dir, -A.tensor, A.size, A.dims, A.division, A.ifZ2)

CuArray{T}(A::U1Array) where {T} = U1Array(A.qn, A.dir, CuArray{T}(A.tensor), A.size, A.dims, A.division, A.ifZ2)
Array{T}(A::U1Array) where {T} = U1Array(A.qn, A.dir, Array{T}(A.tensor), A.size, A.dims, A.division, A.ifZ2)

CuArray(A::U1Array) = U1Array(A.qn, A.dir, CuArray(A.tensor), A.size, A.dims, A.division, A.ifZ2)
Array(A::U1Array) = U1Array(A.qn, A.dir, Array(A.tensor), A.size, A.dims, A.division, A.ifZ2)

function ≈(A::U1Array{TA,NA}, B::U1Array{TB,NB}) where {TA,NA,TB,NB}
    NA != NB && throw(Base.error("$A and $B have different dimensions"))
    A.dir != B.dir && throw(Base.error("$A and $B have different directions"))
    @assert A.qn == B.qn
    A.tensor ≈ B.tensor
end

function ==(A::U1Array{TA,NA}, B::U1Array{TB,NB}) where {TA,NA,TB,NB}
    NA != NB && throw(Base.error("$A and $B have different dimensions"))
    A.dir != B.dir && throw(Base.error("$A and $B have different directions"))
    @assert A.qn == B.qn
    A.tensor == B.tensor
end

function show(::IOBuffer, A::U1Array)
    println("particle number: \n", A.qn)
    println("direction: \n", A.dir)
    println("dims: \n", A.dims)
    println("ifZ2: \n", A.ifZ2)
    # div = blockdiv(A.dims)
    # tensor = [reshape(A.tensor[div[i]], A.dims[i]...) for i in 1:length(A.dims)]
    println("tensor: \n", A.tensor)
end

function sqrt(A::U1Array)
    blocklen = map(x->x[1], A.dims)
    bdiv = [sum(blocklen[1 : i - 1]) + 1 : sum(blocklen[1 : i]) for i in 1:length(blocklen)]
    tensor = vcat([vec(sqrt.(@view(A.tensor[bdiv[i]]))) for i in 1:length(bdiv)]...)
    U1Array(A.qn, A.dir, tensor, A.size, A.dims, A.division, A.ifZ2)
end
broadcasted(sqrt, A::U1Array) = sqrt(A)

function dot(A::U1Array, B::U1Array) 
    @assert A.qn == B.qn
    dot(A.tensor, B.tensor)
end

# # for ' in ACCtoALAR of TeneT
function adjoint(A::U1Array{T,N}) where {T,N}
    div = A.division 
    qn = map(x->x[[div+1:end;1:div]], A.qn)
    p = sortperm(qn)
    dims = A.dims
    bdiv = blockdiv(dims) 
    tensor = vcat([vec(adjoint(reshape(@view(A.tensor[bdiv[i]]), prod(dims[i][1:div]), prod(dims[i][div+1:end])))) for i in 1:length(bdiv)][p]...)
    dims = map(x -> x[[div+1:end;1:div]], A.dims)
    U1Array(qn[p], -A.dir[[div+1:end;1:div]], tensor, A.size[[div+1:end;1:div]], dims[p], N - div, A.ifZ2)
end

function diag(A::U1Array)
    tensor = A.tensor
    dims = A.dims
    bdiv = blockdiv(dims) 
    tensor = [reshape(tensor[bdiv[i]], dims[i]...) for i in 1:length(bdiv)]
    CUDA.@allowscalar collect(Iterators.flatten(diag.(tensor)))
end

# # for leftorth and rightorth compatibility
function Diagonal(A::U1Array{T,2}) where {T}
    atype = _arraytype(A.tensor)
    blocklen = map(x->x[1], A.dims)
    bdiv = [sum(blocklen[1 : i - 1]) + 1 : sum(blocklen[1 : i]) for i in 1:length(blocklen)]
    tensor = CUDA.@allowscalar atype(vcat([vec(diagm(A.tensor[bdiv[i]])) for i in 1:length(bdiv)]...))
    U1Array(A.qn, A.dir, tensor, A.size, A.dims, A.division, A.ifZ2)
end

function invDiagU1Matrix(A::U1Array{T,2}) where {T}
    Adims = A.dims   
    Abdiv = blockdiv(Adims)
    Atensor = vcat([ vec(diagm(1.0 ./diag(reshape(@view(A.tensor[Abdiv[i]]), Adims[i][1], Adims[i][2]))))  for i in 1:length(Abdiv)]...)
    U1Array(A.qn, A.dir, Atensor, A.size, A.dims, A.division, A.ifZ2)
end

function sqrtDiagU1Matrix(A::U1Array{T,2}) where {T}
    Adims = A.dims   
    Abdiv = blockdiv(Adims)
    Atensor = vcat([ vec(diagm( sqrt.(diag(reshape(@view(A.tensor[Abdiv[i]]), Adims[i][1], Adims[i][2])) )))  for i in 1:length(Abdiv)]...)
    U1Array(A.qn, A.dir, Atensor, A.size, A.dims, A.division, A.ifZ2)
end
