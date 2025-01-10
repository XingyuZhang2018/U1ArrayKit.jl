safesign(x::Number) = iszero(x) ? one(x) : sign(x)

"""
    qrpos(A)

Returns a QR decomposition, i.e. an isometric `Q` and upper triangular `R` matrix, where `R`
is guaranteed to have positive diagonal elements.
"""
qrpos(A) = qrpos!(copy(A))
function qrpos!(A)
    mattype = _mattype(A)
    F = qr!(mattype(A))
    Q = mattype(F.Q)
    R = F.R
    phases = safesign.(diag(R))
    Q .= Q * Diagonal(phases)
    R .= Diagonal(conj.(phases)) * R
    return Q, R
end

function qrpos!(A::Union{U1Array{T,2},U1Array{T,3}}) where {T}
    Qqn, Rqn, blockidims = [Vector{Vector{Int}}() for _ in 1:3]
    blockjdims = Vector{Int}()
    indexs = Vector()
    Adims = A.dims
    Aqn = A.qn
    Adir = A.dir
    Adiv = A.division
    Asize = A.size
    Abdiv = blockdiv(Adims)
    Atensor = [reshape(@view(A.tensor[Abdiv[i]]), prod(Adims[i][1:Adiv]), prod(Adims[i][Adiv+1:end])) for i in 1:length(Abdiv)]

    qs = A.ifZ2 ? map(x->sum(x[Adiv+1:end]) % 2, A.qn) : map(x->sum(x[Adiv+1:end] .* Adir[Adiv+1:end]), A.qn)
    for q in unique!(qs)
        u1blockQRinfo!(Qqn, Rqn, indexs, blockidims, blockjdims, Aqn, Adiv, Adir, size.(Atensor), q, A.ifZ2)
    end
    Qtensorlen = sum([sum(blockidims[i]) * blockjdims[i] for i in 1:length(blockidims)])
    Qtensor = typeof(A.tensor) <: Array ? zeros(T, Qtensorlen) : CUDA.zeros(T, Qtensorlen)
    Rtensorlen = sum([blockjdims[i] ^ 2 for i in 1:length(blockjdims)])
    Rtensor = typeof(A.tensor) <: Array ? zeros(T, Rtensorlen) : CUDA.zeros(T, Rtensorlen)

    pp = indexin(Qqn, Aqn)
    Qbdiv = blockdiv(Adims)[pp]
    divs = [length(indexs[i]) for i in 1:length(indexs)]
    bdivind = [sum(divs[1:i-1]) + 1 : sum(divs[1:i]) for i in 1:length(indexs)]

    p = sortperm(Rqn)
    pp = indexin(Rqn, Rqn[p])
    Rdims = [[blockjdims[i], blockjdims[i]] for i in 1:length(blockjdims)]
    Rbdiv = blockdiv(Rdims[p])[pp]

    for i in 1:length(indexs)
        u1blockQR!(Qtensor, Rtensor, Atensor, indexs[i], blockidims[i], Qbdiv[bdivind[i]], Rbdiv[i])
    end
    U1Array(Aqn, Adir, Qtensor, Asize, Adims, Adiv, A.ifZ2), U1Array(Rqn[p], [-A.dir[end], A.dir[end]], Rtensor, (Asize[end], Asize[end]), Rdims[p], 1, A.ifZ2)
end

function u1blockQRinfo!(Qqn, Rqn, indexs, blockidims, blockjdims, Aqn, Adiv, Adir, Atensorsize, q, ifZ2)
    ind_A = ifZ2 ? [sum(Aqn[Adiv+1:end]) % 2 == q for Aqn in Aqn] : [sum(Aqn[Adiv+1:end] .* Adir[Adiv+1:end]) == q for Aqn in Aqn]
    matrix_j = unique!(map(x->x[Adiv+1:end], Aqn[ind_A]))
    matrix_i = unique!(map(x->x[1:Adiv], Aqn[ind_A]))

    index = indexin([[i; matrix_j[1]] for i in matrix_i], Aqn)
    push!(indexs, index)
    push!(blockidims, [Atensorsize[i][1] for i in index])
    push!(blockjdims, Atensorsize[index[1]][2])

    for i in 1:length(matrix_i)
        push!(Qqn, [matrix_i[i]; matrix_j[1]])
    end
    push!(Rqn, [matrix_j[1]; matrix_j[1]])
end

function u1blockQR!(Qtensor, Rtensor, Atensor, index, blockidims, Qbdiv, Rdiv)
    Amatrix = vcat(Atensor[index]...)
    Q, R = qrpos!(Amatrix)
    for i in 1:length(index)
        idim = sum(blockidims[1:i-1])+1:sum(blockidims[1:i])
        Qtensor[Qbdiv[i]] .= vec(@view(Q[idim, :]))
    end
    Rtensor[Rdiv] .= vec(R)
end

function sortqn(A::U1Array)
    qn = A.qn
    p = sortperm(qn)
    qn[p] == qn && return A
    dims = A.dims
    div = blockdiv(dims)
    tensor = vcat([vec(reshape(@view(A.tensor[div[i]]), dims[i]...)) for i in 1:length(div)][p]...)
    U1Array(qn[p], A.dir, tensor, A.size, dims[p], A.division, A.ifZ2)
end

"""
    qrpos!(A::U1Array{T,N}; middledir=1) where {T,N}

QR decomposition for U1Array, return Q, R
middledir is the direction of middle index, 1 for Q → R, -1 for Q ← R
"""
function qrpos!(A::U1Array{T,N}; middledir::Int = 1) where {T,N}
    Qqn, Rqn, Qdims, Rdims, blockidims, blockjdims = [Vector{Vector{Int}}() for _ in 1:6]
    indexs = Vector()
    Adims = A.dims
    Aqn = A.qn
    Adir = A.dir
    Adiv = A.division
    Asize = A.size
    Abdiv = blockdiv(Adims)
    Atensor = [reshape(@view(A.tensor[Abdiv[i]]), prod(Adims[i][1:Adiv]), prod(Adims[i][Adiv+1:end])) for i in 1:length(Abdiv)]

    qs = A.ifZ2 ? 
         sort!(map(x->sum(x[Adiv+1:end]) % 2, A.qn)) : 
         sort!(map(x->sum(x[Adiv+1:end] .* Adir[Adiv+1:end]), A.qn))
    for q in unique!(qs)
        u1blockQRinfo!(Qqn, Rqn, Qdims, Rdims, indexs, blockidims, blockjdims, Aqn, Adims, Adiv, Adir, size.(Atensor), q, A.ifZ2)
    end
    Qtensorlen = [sum(blockidims[i]) * min(sum(blockidims[i]), sum(blockjdims[i])) for i in 1:length(indexs)]
    Qtensor = typeof(A.tensor) <: Array ? zeros(T, sum(Qtensorlen)) : CUDA.zeros(T, sum(Qtensorlen))

    Rtensorlen = [min(sum(blockidims[i]), sum(blockjdims[i])) * sum(blockjdims[i]) for i in 1:length(indexs)]
    Rtensor = typeof(A.tensor) <: Array ? zeros(T, sum(Rtensorlen)) : CUDA.zeros(T, sum(Rtensorlen))

    Qbdiv = blockdiv(Qdims)
    Qdivs = length.(blockidims)
    Qbdivind = [sum(Qdivs[1:i-1]) + 1 : sum(Qdivs[1:i]) for i in 1:length(indexs)]

    Rbdiv = blockdiv(Rdims)
    Rdivs = length.(blockjdims)
    Rbdivind = [sum(Rdivs[1:i-1]) + 1 : sum(Rdivs[1:i]) for i in 1:length(indexs)]

    for i in 1:length(indexs)
        u1blockQR!(Qtensor, Rtensor, Atensor, indexs[i], blockidims[i], blockjdims[i], Qbdiv[Qbdivind[i]], Rbdiv[Rbdivind[i]])
    end

    middledim = min(prod(Asize[1:Adiv]), prod(Asize[Adiv+1:end]))

    return sortqn(U1Array(Qqn, [Adir[1:Adiv]...; middledir], Qtensor, (Asize[1:Adiv]..., middledim), Qdims, Adiv, A.ifZ2)), 
    sortqn(U1Array(Rqn, [-middledir, Adir[Adiv+1:end]...], Rtensor, (middledim, Asize[Adiv+1:end]...), Rdims, 1, A.ifZ2))
end

function u1blockQRinfo!(Qqn, Rqn, Qdims, Rdims, indexs, blockidims, blockjdims, Aqn, Adims, Adiv, Adir, Atensorsize, q, ifZ2)
    ind_A = ifZ2 ? [sum(Aqn[Adiv+1:end]) % 2 == q for Aqn in Aqn] : [sum(Aqn[Adiv+1:end] .* Adir[Adiv+1:end]) == q for Aqn in Aqn]
    
    matrix_i = unique!(sort!(map(x->x[1:Adiv], Aqn[ind_A])))
    matrix_j = unique!(sort!(map(x->x[Adiv+1:end], Aqn[ind_A])))

    index = indexin([[i; j] for i in matrix_i, j in matrix_j], Aqn) 
    push!(indexs, index)
    push!(blockidims, [Atensorsize[i][1] for i in index[:, 1]])
    push!(blockjdims, [Atensorsize[i][2] for i in index[1, :]])
    middledim = min(sum([prod(Adims[i][1:Adiv]) for i in index[:, 1]]), sum([prod(Adims[i][Adiv+1:end]) for i in index[1, :]]))
    [push!(Qdims, [Adims[i][1:Adiv]...; middledim]) for i in index[:, 1]]
    [push!(Rdims, [middledim; Adims[i][Adiv+1:end]...]) for i in index[1, :]]

    for i in matrix_i
        push!(Qqn, [i; q])
    end
    for j in matrix_j
        push!(Rqn, [q; j])
    end
end

function u1blockQR!(Qtensor, Rtensor, Atensor, index, blockidims, blockjdims, Qbdiv, Rbdiv)
    Amatrix = _arraytype(Qtensor) <: Array ? zeros(eltype(Qtensor), sum(blockidims), sum(blockjdims)) : CUDA.zeros(eltype(Qtensor), sum(blockidims), sum(blockjdims))
    for i in 1:size(index, 1), j in 1:size(index, 2)
        index[i, j] !== nothing && (Amatrix[sum(blockidims[1:i-1])+1:sum(blockidims[1:i]), sum(blockjdims[1:j-1])+1:sum(blockjdims[1:j])] .= Atensor[index[i, j]])
    end

    Qmatrix, Rmatrix = qrpos!(Amatrix)

    for i in 1:length(blockidims)
        idim = sum(blockidims[1:i-1])+1:sum(blockidims[1:i])
        Qtensor[Qbdiv[i]] .= vec(@view(Qmatrix[idim, :]))
    end
    for j in 1:length(blockjdims)
        jdim = sum(blockjdims[1:j-1])+1:sum(blockjdims[1:j])
        Rtensor[Rbdiv[j]] .= vec(@view(Rmatrix[:, jdim]))
    end
end

"""
    lqpos(A)

Returns a LQ decomposition, i.e. a lower triangular `L` and isometric `Q` matrix, where `L`
is guaranteed to have positive diagonal elements.
"""
lqpos(A) = lqpos!(copy(A))
function lqpos!(A)
    mattype = _mattype(A)
    F = qr!(mattype(A'))
    Q = mattype(mattype(F.Q)')
    L = mattype(F.R')
    phases = safesign.(diag(L))
    Q .= Diagonal(phases) * Q
    L .= L * Diagonal(conj!(phases))
    return L, Q
end

function lqpos!(A::U1Array{T,N}) where {T, N}
    Lqn, Qqn, blockjdims = [Vector{Vector{Int}}() for _ in 1:3]
    blockidims = Vector{Int}()
    indexs = Vector()
    Adims = A.dims
    Aqn = A.qn
    Adir = A.dir
    Adiv = A.division
    Asize = A.size
    Abdiv = blockdiv(Adims)
    Atensor = [reshape(@view(A.tensor[Abdiv[i]]), prod(Adims[i][1:Adiv]), prod(Adims[i][Adiv+1:end])) for i in 1:length(Abdiv)]

    qs = A.ifZ2 ? map(x->x[1] % 2, Aqn) : map(x->x[1] * Adir[1], Aqn)
    for q in unique!(qs)
        u1blockLQinfo!(Lqn, Qqn, indexs, blockidims, blockjdims, Aqn, Adiv, Adir, size.(Atensor), q, A.ifZ2)
    end

    Qtensorlen = sum([blockidims[i] * sum(blockjdims[i]) for i in 1:length(blockidims)])
    Qtensor = typeof(A.tensor) <: Array ? zeros(T, Qtensorlen) : CUDA.zeros(T, Qtensorlen)
    Ltensorlen = sum([blockidims[i] ^ 2 for i in 1:length(blockidims)])
    Ltensor = typeof(A.tensor) <: Array ? zeros(T, Ltensorlen) : CUDA.zeros(T, Ltensorlen)

    pp = indexin(Qqn, Aqn)
    Qbdiv = blockdiv(Adims)[pp]
    divs = [length(indexs[i]) for i in 1:length(indexs)]
    bdivind = [sum(divs[1:i-1]) + 1 : sum(divs[1:i]) for i in 1:length(indexs)]

    p = sortperm(Lqn)
    pp = indexin(Lqn, Lqn[p])
    Ldims = [[blockidims[i], blockidims[i]] for i in 1:length(blockidims)]
    Lbdiv = blockdiv(Ldims[p])[pp]

    for i in 1:length(indexs)
        u1blockLQ!(Ltensor, Qtensor, Atensor, indexs[i], blockjdims[i], Qbdiv[bdivind[i]], Lbdiv[i])
    end

    U1Array(Lqn[p], [A.dir[1], -A.dir[1]], Ltensor, (Asize[1], Asize[1]), Ldims[p], 1, A.ifZ2), U1Array(Aqn, A.dir, Qtensor, Asize, Adims, A.division, A.ifZ2)
end

function u1blockLQinfo!(Lqn, Qqn, indexs, blockidims, blockjdims, Aqn, Adiv, Adir, Atensorsize, q, ifZ2)
    ind_A = ifZ2 ?  [sum(Aqn[1]) % 2 == q for Aqn in Aqn] : [sum(Aqn[1] .* Adir[1]) == q for Aqn in Aqn]
    matrix_j = unique!(map(x->x[Adiv+1:end], Aqn[ind_A]))
    matrix_i = unique!(map(x->x[1], Aqn[ind_A]))

    index = indexin([[matrix_i[1]; j] for j in matrix_j], Aqn)
    push!(indexs, index)
    push!(blockidims, Atensorsize[index[1]][1])
    push!(blockjdims, [Atensorsize[i][2] for i in index])

    for j in 1:length(matrix_j)
        push!(Qqn, [matrix_i[1]; matrix_j[j]])
    end
    push!(Lqn, [matrix_i[1]; matrix_i[1]])
end

function u1blockLQ!(Ltensor, Qtensor, Atensor, index, blockidims, Qbdiv, Ldiv)
    Amatrix = hcat(Atensor[index]...)
    L, Q = lqpos!(Amatrix)

    Ltensor[Ldiv] .= vec(L)
    for j in 1:length(index)
        jdim = sum(blockidims[1:j-1])+1:sum(blockidims[1:j])
        Qtensor[Qbdiv[j]] .= vec(@view(Q[:, jdim]))
    end
end


# only for U1 Matrix
svd(A::U1Array; kwargs...) = svd!(copy(A); kwargs...)
function svd!(A::U1Array{T,2}; trunc::Int = -1) where {T}
    tensor = A.tensor
    qn = A.qn
    div = A.division
    atype = _arraytype(A.tensor)

    Adims = A.dims
    Abdiv = blockdiv(Adims)
    tensor = [reshape(@view(A.tensor[Abdiv[i]]), prod(Adims[i][1:div]), prod(Adims[i][div+1:end])) for i in 1:length(Abdiv)]

    Utensor = Vector{atype{T}}()
    Stensor = Vector{atype{T}}()
    Vtensor = Vector{atype{T}}()
    @inbounds @simd for t in tensor
        F = svd!(t)
        push!(Utensor, F.U)
        push!(Stensor, F.S)
        push!(Vtensor, F.V)
    end
    if trunc != -1
        Sort = sort(abs.(vcat(Stensor...)); rev=true)
        minS = max(Sort[trunc], 1e-30)
        ind = [abs.(Stensor[i]) .>= minS for i in 1:length(Stensor)]
        for i in 1:length(qn)
            Utensor[i] = Utensor[i][:, ind[i]]
            Stensor[i] = Stensor[i][ind[i]]
            Vtensor[i] = Vtensor[i][:, ind[i]]
        end
        deleind = sum.(ind) .== 0
        deleteat!(qn, deleind)
        deleteat!(Utensor, deleind)
        deleteat!(Stensor, deleind)
        deleteat!(Vtensor, deleind)
    end

    Udims = map(x -> collect(size(x)), Utensor)
    Sdims = map(x -> [length(x), length(x)], Stensor)
    Vdims = map(x -> collect(size(x)), Vtensor)
    Asize = A.size
    sm = min(Asize...)
    Utensor = vcat(map(vec, Utensor)...)
    Stensor = vcat(map(vec, Stensor)...)
    Vtensor = vcat(map(vec, Vtensor)...)
    U1Array(qn, A.dir, Utensor, (Asize[1], sm), Udims, div, A.ifZ2), U1Array(qn, A.dir, Stensor, (sm, sm), Sdims, div, A.ifZ2), U1Array(qn, A.dir, Vtensor, (Asize[2], sm),  Vdims, div, A.ifZ2)
end

"""
    svd!(A::U1Array{T,N}; trunc::Int = -1, middledir::Int = 1) where {T,N}

SVD decomposition for U1Array, return U, S, V
trunc is the truncation dimension, -1 for no truncation
middledir is the direction of middle index, 1 for U → S → V, -1 for U ← S ← V
"""
function svd!(A::U1Array{T,N}; trunc::Int = -1, middledir::Int = 1, ifSU = false) where {T,N}
    Uqn, Sqn, Vqn, Udims, Sdims, Vdims, blockidims, blockjdims = [Vector{Vector{Int}}() for _ in 1:8]
    indexs = Vector()
    Adims = A.dims
    Aqn = A.qn
    Adir = A.dir
    Adiv = A.division
    Asize = A.size
    Abdiv = blockdiv(Adims)
    Atensor = [reshape(@view(A.tensor[Abdiv[i]]), prod(Adims[i][1:Adiv]), prod(Adims[i][Adiv+1:end])) for i in 1:length(Abdiv)]

    qs = A.ifZ2 ? 
         sort!(map(x->sum(x[Adiv+1:end]) % 2, A.qn)) : 
         sort!(map(x->sum(x[Adiv+1:end] .* Adir[Adiv+1:end]), A.qn))
    for q in unique!(qs)
        u1blockSVDinfo!(Uqn, Sqn, Vqn, Udims, Sdims, Vdims, indexs, blockidims, blockjdims, Aqn, Adims, Adiv, Adir, size.(Atensor), q, A.ifZ2)
    end

    qlen = length(indexs)

    Utensorlen = [sum(blockidims[i]) * min(sum(blockidims[i]), sum(blockjdims[i])) for i in 1:qlen]
    Utensor = typeof(A.tensor) <: Array ? zeros(T, sum(Utensorlen)) : CUDA.zeros(T, sum(Utensorlen))

    Stensorlen = [min(sum(blockidims[i]), sum(blockjdims[i])) for i in 1:qlen]
    Stensor = typeof(A.tensor) <: Array ? zeros(T, sum(Stensorlen)) : CUDA.zeros(T, sum(Stensorlen))

    Vtensorlen = [min(sum(blockidims[i]), sum(blockjdims[i])) * sum(blockjdims[i]) for i in 1:qlen]
    Vtensor = typeof(A.tensor) <: Array ? zeros(T, sum(Vtensorlen)) : CUDA.zeros(T, sum(Vtensorlen))

    Ubdiv = blockdiv(Udims)
    Udivs = length.(blockidims)
    Ubdivind = [sum(Udivs[1:i-1]) + 1 : sum(Udivs[1:i]) for i in 1:qlen]

    blocklen = map(x->x[1], Sdims)
    Sbdiv = [sum(blocklen[1:i-1]) + 1 : sum(blocklen[1:i]) for i in 1:length(blocklen)]

    Vbdiv = blockdiv(Vdims)
    Vdivs = length.(blockjdims)
    Vbdivind = [sum(Vdivs[1:i-1]) + 1 : sum(Vdivs[1:i]) for i in 1:qlen]

    for i in 1:qlen
        u1blockSVD!(Utensor, Stensor, Vtensor, Atensor, indexs[i], blockidims[i], blockjdims[i], Ubdiv[Ubdivind[i]], Sbdiv[i], Vbdiv[Vbdivind[i]])
    end

    middledim = min(prod(Asize[1:Adiv]), prod(Asize[Adiv+1:end]), sum([dims[1] for dims in Sdims]))

    if trunc != -1 && trunc < middledim
        middledim = trunc
        if ifSU # for the moment, only for simple update Z2 symmetry case
            Slen = length(Stensor)
            evenhalf = ceil(Int, trunc / 2)
            oddhalf = trunc - evenhalf
            ind = [[true for _ in 1:Slen÷2] for _ in 1:2]
            ind[1][1:evenhalf] .= false
            ind[2][1:oddhalf] .= false
        else
            Sort = sort(abs.(Stensor); rev=true)
            minS = max(Sort[trunc], 1e-30)
            ind = [abs.(Stensor[Sbdiv[i]]) .< minS for i in 1:qlen]
        end

        deleind = (1:qlen)[sum.(ind) .!== 0]

        Udeleind = [[[block[end]-sum(ind[i])*dims+1 : block[end]...] for (block,dims) in zip(Ubdiv[Ubdivind[i]],blockidims[i])] for i in deleind]
        Udeleind = vcat(vcat(Udeleind...)...)
        deleteat!(Utensor, Udeleind)
        [[dims .= [dims[1:end-1]; length(ind[i]) - sum(ind[i])] for dims in Udims[Ubdivind[i]]] for i in deleind]

        Sdeleind = vcat(ind...)
        deleteat!(Stensor, Sdeleind)
        [[Sdims[i] .= [length(ind[i]) - sum(ind[i])]] for i in deleind]

        Vdeleind = [[[block[end]-sum(ind[i])*dims+1 : block[end]...] for (block,dims) in zip(Vbdiv[Vbdivind[i]],blockjdims[i])] for i in deleind]
        Vdeleind = vcat(vcat(Vdeleind...)...)
        deleteat!(Vtensor, Vdeleind)
        [[dims .= [dims[1:end-1]; length(ind[i]) - sum(ind[i])] for dims in Vdims[Vbdivind[i]]] for i in deleind]

        Sqdeleind = (1:qlen)[sum.(ind) .== length.(ind)]
        deleteat!(Sqn, Sqdeleind)
        deleteat!(Sdims, Sqdeleind)

        Uqdeleind = vcat(Ubdivind[Sqdeleind]...)
        deleteat!(Uqn, Uqdeleind)
        deleteat!(Udims, Uqdeleind)

        Vqdeleind = vcat(Vbdivind[Sqdeleind]...)
        deleteat!(Vqn, Vqdeleind)
        deleteat!(Vdims, Vqdeleind)
    end
    
    return U1Array(Uqn, [Adir[1:Adiv]...; middledir], Utensor, (Asize[1:Adiv]..., middledim), Udims, Adiv, A.ifZ2), 
    U1Array(Sqn, [-middledir; middledir], Stensor, (middledim, middledim), Sdims, 1, A.ifZ2),
    U1Array(Vqn, [-Adir[Adiv+1:end]..., middledir], Vtensor, (Asize[Adiv+1:end]..., middledim), Vdims, N-Adiv, A.ifZ2)
end

function u1blockSVDinfo!(Uqn, Sqn, Vqn, Udims, Sdims, Vdims, indexs, blockidims, blockjdims, Aqn, Adims, Adiv, Adir, Atensorsize, q, ifZ2)
    ind_A = ifZ2 ? [sum(Aqn[Adiv+1:end]) % 2 == q for Aqn in Aqn] : [sum(Aqn[Adiv+1:end] .* Adir[Adiv+1:end]) == q for Aqn in Aqn]
    
    matrix_i = unique!(sort!(map(x->x[1:Adiv], Aqn[ind_A])))
    matrix_j = unique!(sort!(map(x->x[Adiv+1:end], Aqn[ind_A])))

    index = indexin([[i; j] for i in matrix_i, j in matrix_j], Aqn) 
    push!(indexs, index)
    push!(blockidims, [Atensorsize[i][1] for i in index[:, 1]])
    push!(blockjdims, [Atensorsize[i][2] for i in index[1, :]])
    middledim = min(sum([prod(Adims[i][1:Adiv]) for i in index[:, 1]]), sum([prod(Adims[i][Adiv+1:end]) for i in index[1, :]]))
    [push!(Udims, [Adims[i][1:Adiv]...; middledim]) for i in index[:, 1]]
    push!(Sdims, [middledim; middledim])
    [push!(Vdims, [Adims[i][Adiv+1:end]...; middledim]) for i in index[1, :]]

    for i in matrix_i
        push!(Uqn, [i; q])
    end
    push!(Sqn, [q; q])
    for j in matrix_j
        push!(Vqn, [j; q])
    end
end

function u1blockSVD!(Utensor, Stensor, Vtensor, Atensor, index, blockidims, blockjdims, Ubdiv, Sbdiv, Vbdiv)
    Amatrix = _arraytype(Utensor) <: Array ? zeros(eltype(Utensor), sum(blockidims), sum(blockjdims)) : CUDA.zeros(eltype(Utensor), sum(blockidims), sum(blockjdims))
    for i in 1:size(index, 1), j in 1:size(index, 2)
        index[i, j] !== nothing && (Amatrix[sum(blockidims[1:i-1])+1:sum(blockidims[1:i]), sum(blockjdims[1:j-1])+1:sum(blockjdims[1:j])] .= Atensor[index[i, j]])
    end

    Umatrix, Svector, Vmatrix = svd!(Amatrix)

    for i in 1:length(blockidims)
        idim = sum(blockidims[1:i-1])+1:sum(blockidims[1:i])
        Utensor[Ubdiv[i]] .= vec(@view(Umatrix[idim, :]))
    end
    Stensor[Sbdiv] .= vec(@view(Svector[:]))
    for j in 1:length(blockjdims)
        jdim = sum(blockjdims[1:j-1])+1:sum(blockjdims[1:j])
        Vtensor[Vbdiv[j]] .= vec(@view(Vmatrix[jdim, :]))
    end
end