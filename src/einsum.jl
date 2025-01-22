
function permute_perm(dims, perm)
    len = map(prod, dims)
    vcat([sum(len[1:i-1]) .+ vec(permutedims(reshape(collect(1:len[i]), dims[i]...), perm)) for i in 1:length(dims)]...)
end

# only for OMEinsum binary permutedims before reshape
permutedims(A::U1Array, perm) = tensorpermute(A, perm)
function tensorpermute(A::U1Array{T, N}, perm) where {T <: Number, N}
    length(perm) == 0 && return copy(A)
    dims = A.dims
    qn = map(x -> x[collect(perm)], A.qn)
    p = sortperm(qn)
    div = blockdiv(dims)
    tensor = _arraytype(A.tensor){T}(vcat([vec(permutedims(reshape(@view(A.tensor[div[i]]), dims[i]...), perm)) for i in 1:length(div)][p]...))
    dims = map(x -> x[collect(perm)], dims)
    U1Array(qn[p], A.dir[collect(perm)], tensor, A.size[collect(perm)], dims[p], A.division, A.ifZ2)
end

reshape(A::U1Array, s::Tuple{Vararg{Int}}) = reshape(A, s...)
function reshape(A::U1Array{T,N}, s::Int...) where {T <: Number,N}
    div = 1
    if length(s) < N
        sizeA = size(A)
        p = sizeA[1]
        while p != s[1]
            div += 1
            p *= sizeA[div]
        end
        return U1Array(A.qn, A.dir, A.tensor, A.size, A.dims, div, A.ifZ2)
    else
        return A
    end
end

"""
    *(A::U1Array{TA,NA}, B::U1Array{TB,NB}) where {TA,TB,NA,NB}

core code for U1Array product
"""
function *(A::U1Array{T, NA}, B::U1Array{T, NB}) where {T, NA, NB}
    atype = typeof(A.tensor) <: CuArray ? CuArray : Array
    qn, dims, blockidims, blockjdims, blockkdims = [Vector{Vector{Int}}() for _ in 1:7]
    Aindexs, Bindexs =  [Vector() for _ in 1:2]
    Aqn, Bqn = A.qn, B.qn
    Adims, Bdims = A.dims, B.dims
    Abdiv = blockdiv(Adims)
    Bbdiv = blockdiv(Bdims)
    Adiv, Bdiv = A.division, B.division
    Atensor = [reshape(@view(A.tensor[Abdiv[i]]), prod(Adims[i][1:Adiv]), prod(Adims[i][Adiv+1:end])) for i in 1:length(Abdiv)]
    Btensor = [reshape(@view(B.tensor[Bbdiv[i]]), prod(Bdims[i][1:Bdiv]), prod(Bdims[i][Bdiv+1:end])) for i in 1:length(Bbdiv)]
    
    if !(A.ifZ2 && B.ifZ2)
        timesAdir = getdir(A)[Adiv+1:end]
        timesBdir = getdir(B)[1:Bdiv]
        sum(timesAdir .+ timesBdir) !== 0 && throw(Base.error("U1Array product: out and in direction not match, expect: $(-timesAdir), got: $(timesBdir)"))
    end

    qs = A.ifZ2 ? map(qn -> sum(qn[Adiv+1:end]) % 2, Aqn) : map(qn -> sum(qn[Adiv+1:end] .* A.dir[Adiv+1:end]), Aqn)
    for q in unique!(qs)
        timesinfo!(qn, dims, Aindexs, Bindexs, blockidims, blockjdims, blockkdims,
                        Aqn, Adiv, A.dir, size.(Atensor), Adims, Bqn, Bdiv, size.(Btensor), Bdims, q, A.ifZ2)
    end
    tensorlen = sum([sum(blockidims[i]) * sum(blockkdims[i]) for i in 1:length(blockidims)])
    tensor = atype <: Array ? zeros(T, tensorlen) : CUDA.zeros(T, tensorlen)

    p = sortperm(qn)
    pp = indexin(qn, qn[p])
    bdiv = blockdiv(dims[p])[pp]
    divs = [size(Aindexs[i], 1) * size(Bindexs[i], 2) for i in 1:length(Aindexs)]
    bdivind = [sum(divs[1:i-1]) + 1 : sum(divs[1:i]) for i in 1:length(Aindexs)]
    for i in 1:length(Aindexs)
        u1blocktimes!(tensor, Aindexs[i], Bindexs[i], blockidims[i], blockjdims[i], blockkdims[i], bdiv[bdivind[i]], Atensor, Btensor)
    end
    qn == [[]] && return Array(tensor)[]
    U1Array(qn[p], [A.dir[1:Adiv]..., B.dir[Bdiv+1:end]...], tensor, (size(A)[1:Adiv]..., size(B)[Bdiv+1:end]...), dims[p], Adiv, A.ifZ2)
end

function no_nothing_col(index)
    indexcol = Int[]
    for i in 1:size(index,1)
        for j in 1:size(index,2)
            if index[i, j] !== nothing
                push!(indexcol, index[i, j])
                break
            end
        end
    end
    indexcol
end

function no_nothing_row(index)
    indexrow = Int[]
    for j in 1:size(index,2)
        for i in 1:size(index,1)
            if index[i, j] !== nothing
                push!(indexrow, index[i, j])
                break
            end
        end
    end
    indexrow
end

"""
    u1blocktimes!(qn, tensor, A, B, q)

fill into different quantum number,  then dispatch to result tensor after product
"""
function timesinfo!(qn, dims, Aindexs, Bindexs, blockidims, blockjdims, blockkdims,
                    Aqn, Adiv, Adir, Atensorsize, Adims, Bqn, Bdiv, Btensorsize, Bdims, q, ifZ2)
    @inbounds @views begin
        ind_A = ifZ2 ? [sum(Aqn[Adiv+1:end]) % 2 == q for Aqn in Aqn] : [sum(Aqn[Adiv+1:end] .* Adir[Adiv+1:end]) == q for Aqn in Aqn]
        matrix_j = intersect!(map(x->x[Adiv+1:end], Aqn[ind_A]), map(x->x[1:Bdiv], Bqn))
        ind_A = [Aqn[Adiv+1:end] in matrix_j for Aqn in Aqn]
        matrix_i = unique!(map(x->x[1:Adiv], Aqn[ind_A]))
        ind_B = [Bqn[1:Bdiv] in matrix_j for Bqn in Bqn]
        sum(ind_B) == 0 && return
        matrix_k = unique!(map(x->x[Bdiv+1:end], Bqn[ind_B]))
    end

    Aindex = indexin([[i; j] for i in matrix_i, j in matrix_j], Aqn)
    Bindex = indexin([[j; k] for j in matrix_j, k in matrix_k], Bqn)
    push!(Aindexs, Aindex)
    push!(Bindexs, Bindex)

    if nothing in Aindex
        indexcol = no_nothing_col(Aindex)
        indexrow = no_nothing_row(Aindex)
    else
        indexcol = @view Aindex[:, 1]
        indexrow = @view Aindex[1, :]
    end

    oriblockidims = map(ind -> Adims[ind][1:Adiv], indexcol)
    push!(blockidims, map(ind -> Atensorsize[ind][1], indexcol))
    push!(blockjdims, map(ind -> Atensorsize[ind][2], indexrow))

    indexrow = nothing in Bindex ? no_nothing_row(Bindex) : (@view Bindex[1, :])
    oriblockkdims = map(ind -> Bdims[ind][Bdiv+1:end], indexrow)
    push!(blockkdims, map(ind -> Btensorsize[ind][2], indexrow))

    for i in 1:length(matrix_i), k in 1:length(matrix_k)
        push!(qn, [matrix_i[i]; matrix_k[k]])
        push!(dims, [oriblockidims[i]; oriblockkdims[k]])
    end
end

function u1blocktimes!(tensor, Aindex, Bindex, blockidims, blockjdims, blockkdims, bdiv, Atensor, Btensor)
    atype = _arraytype(tensor)
    etype = eltype(Atensor[1])
    Amatrix = atype <: Array ? zeros(etype, sum(blockidims), sum(blockjdims)) : CUDA.zeros(etype, sum(blockidims), sum(blockjdims))
    for i in 1:size(Aindex, 1), j in 1:size(Aindex, 2)
        Aindex[i, j] !== nothing && (Amatrix[sum(blockidims[1:i-1])+1:sum(blockidims[1:i]), sum(blockjdims[1:j-1])+1:sum(blockjdims[1:j])] .= Atensor[Aindex[i, j]])
    end 

    Bmatrix = atype <: Array ? zeros(etype, sum(blockjdims), sum(blockkdims)) : CUDA.zeros(etype, sum(blockjdims), sum(blockkdims))
    for j in 1:size(Bindex, 1), k in 1:size(Bindex, 2)
        Bindex[j, k] !== nothing && (Bmatrix[sum(blockjdims[1:j-1])+1:sum(blockjdims[1:j]), sum(blockkdims[1:k-1])+1:sum(blockkdims[1:k])] .= Btensor[Bindex[j, k]])
    end

    Cmatrix = Amatrix * Bmatrix

    for i in 1:size(Aindex, 1), k in 1:size(Bindex, 2)
        idim, kdim = sum(blockidims[1:i-1])+1:sum(blockidims[1:i]), sum(blockkdims[1:k-1])+1:sum(blockkdims[1:k])
        tensor[bdiv[(i-1) * size(Bindex, 2) + k]] .= vec(@view(Cmatrix[idim, kdim]))
    end
end

# # for OMEinsum contract to get number
# # vec(A::U1Array) = A

transpose(A::U1Array) = U1Array(A.qn, A.dir, A.tensor, A.size, A.dims, 0, A.ifZ2)

function tr(A::U1Array{T,2}) where {T}
    qn = A.qn
    tensor = A.tensor
    dims = A.dims
    s = 0.0
    div = blockdiv(dims)
    tensor = [reshape(tensor[div[i]], dims[i]...) for i in 1:length(div)]
    @inbounds @simd for i in 1:length(qn)
        qn[i][1] == qn[i][2] && (s += tr(tensor[i]))
    end
    s
end

# function _compactify!(y, x::U1Array, indexer)
#     x = asArray(Array(x))
#     @inbounds @simd for ci in CartesianIndices(y)
#         y[ci] = x[subindex(indexer, ci.I)]
#     end
#     return y
# end

# broadcasted(*, A::U1Array, B::Base.RefValue) = U1Array(A.qn, A.dir, A.tensor .* B, A.size, A.dims, A.division, A.ifZ2)
# broadcasted(*, B::Base.RefValue, A::U1Array) = U1Array(A.qn, A.dir, A.tensor .* B, A.size, A.dims, A.division, A.ifZ2)

# for ein"abab ->"(A)[]
function dtr(A::U1Array{T,4}) where {T}
    qn = A.qn
    tensor = A.tensor
    dims = A.dims
    s = 0.0
    div = blockdiv(dims)
    tensor = [reshape(tensor[div[i]], dims[i]...) for i in 1:length(div)]
    @inbounds @simd for i in 1:length(qn)
        qn[i][1] == qn[i][3] && qn[i][2] == qn[i][4] && (s += Array(ein"abab ->"(tensor[i]))[])
    end
    s
end
