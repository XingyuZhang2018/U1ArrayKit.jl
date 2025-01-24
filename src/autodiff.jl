# for Zygote compatibility
accum(A::U1Array, B::U1Array...) = +(A, B...)

# patch since it's currently broken otherwise
function ChainRulesCore.rrule(::typeof(Base.typed_hvcat), ::Type{T}, rows::Tuple{Vararg{Int}}, xs::S...) where {T,S}
    y = Base.typed_hvcat(T, rows, xs...)
    function back(ȳ)
        return NoTangent(), NoTangent(), NoTangent(), permutedims(ȳ)...
    end
    return y, back
end

# improves performance compared to default implementation, also avoids errors
# with some complex arrays
function ChainRulesCore.rrule(::typeof(norm), A::AbstractArray{<:Number})
    n = norm(A)
    function back(Δ)
        return NoTangent(), Δ * A / (n + eps(0f0))
    end
    return n, back
end

function ChainRulesCore.rrule(::typeof(Base.sqrt), A::AbstractArray)
    As = Base.sqrt(A)
    function back(dAs)
        dA =  As' \ dAs ./2 
        return NoTangent(), dA
    end
    return As, back
end

@adjoint function reshape(A::U1Array, a::Int...)
    function back(dAr)
        @assert A.qn == dAr.qn
        # s = map(size, A.tensor) 
        # dAtensor = map((x, y) -> reshape(x, y), dAr.tensor, s)
        return U1Array(A.qn, A.dir, dAr.tensor, A.size, A.dims, A.division, A.ifZ2), a...
    end
    return reshape(A, a...), back
end

@adjoint *(A::AbstractSymmetricArray, B::AbstractSymmetricArray) = A * B, dC -> (dC * B', A' * dC)
@adjoint adjoint(A::AbstractSymmetricArray) = adjoint(A), djA -> (adjoint(djA), )
@adjoint conj(A::AbstractSymmetricArray) = conj(A), dA -> (conj(dA), )

# @adjoint conjM(A::AbstractArray) = conjM(A), dA -> (conjM(dA), )

# ChainRulesCore.rrule(::typeof(asArray), sitetype::AbstractSiteType, A::AbstractSymmetricArray) = asArray(sitetype, A), dAt -> (NoTangent(), NoTangent(), asSymmetryArray(dAt, Val(getsymmetry(A)), sitetype; dir = getdir(A)))

# ChainRulesCore.rrule(::typeof(asSymmetryArray), A::AbstractArray, symmetry, sitetype; kwarg...) = asSymmetryArray(A, symmetry, sitetype; kwarg...), dAt -> (NoTangent(), asArray(sitetype, dAt), NoTangent(), NoTangent()...)

function ChainRulesCore.rrule(::typeof(U1Array), qn::Vector{Vector{Int}}, dir::Vector{Int}, tensor::AbstractArray{T}, size::Tuple{Vararg{Int, N}}, dims::Vector{Vector{Int}}, division::Int, ifZ2::Bool) where {T,N}
    function back(dA)
        @assert qn == dA.qn
        return NoTangent(), NoTangent(), NoTangent(), dA.tensor, NoTangent(), NoTangent(), NoTangent(), NoTangent()...
    end
    return U1Array(qn, dir, tensor, size, dims, division, ifZ2), back
end

@adjoint function tr(A::U1Array{T,N}) where {T,N}
    function back(dtrA)
        dA = zero(A)
        atype = _arraytype(A.tensor)
        Abdiv = blockdiv(A.dims)
        for i in 1:length(Abdiv)
            dA.tensor[Abdiv[i]] = vec(atype(Matrix(I,dA.dims[i]...) * dtrA))
        end
        return (dA, )
    end
    tr(A), back
end

# for ein"ijkl,ijkl -> " backward
broadcasted(*, A::U1Array, B::Base.RefValue) = U1Array(A.qn, A.dir, A.tensor .* B, A.size, A.dims, A.division, A.ifZ2)
broadcasted(*, B::Base.RefValue, A::U1Array) = U1Array(A.qn, A.dir, A.tensor .* B, A.size, A.dims, A.division, A.ifZ2)

function ChainRulesCore.rrule(::typeof(dtr), A::U1Array{T,N}) where {T,N}
    function back(dtrA)
        atype = _arraytype(A.tensor)
        Aqn = A.qn
        Adims = A.dims
        dA = zero(A)
        Abdiv = blockdiv(Adims)
        for i in 1:length(Abdiv)
            if Aqn[i][1] == Aqn[i][3] && Aqn[i][2] == Aqn[i][4]
                d1 = Adims[i][1]
                d2 = Adims[i][2]
                dA.tensor[Abdiv[i]] = vec(atype(dtrA * ein"ab, cd -> acbd"(Matrix(I,d1,d1), Matrix(I,d2,d2))))
                # for j = 1:d1, k = 1:d2
                #     dA.tensor[i][j,k,j,k] = dtrA
                # end
            end
        end
        return NoTangent(), atype(deletezeroblock(dA))
    end
    dtr(A), back
end

function ChainRulesCore.rrule(::typeof(qrpos), A::AbstractArray{T,2}) where {T}
    Q, R = qrpos(A)
    function back((dQ, dR))
        M = Array(R * dR' - dQ' * Q)
        dA = (UpperTriangular(R + I * 1e-12) \ (dQ + Q * _arraytype(Q)(Hermitian(M, :L)))' )'
        return NoTangent(), _arraytype(Q)(dA)
    end
    return (Q, R), back
end

function ChainRulesCore.rrule(::typeof(lqpos), A::AbstractArray{T,2}) where {T}
    L, Q = lqpos(A)
    function back((dL, dQ))
        M = Array(L' * dL - dQ * Q')
        dA = LowerTriangular(L + I * 1e-12)' \ (dQ + _arraytype(Q)(Hermitian(M, :L)) * Q)
        return NoTangent(), _arraytype(Q)(dA)
    end
    return (L, Q), back
end

# U1
function ChainRulesCore.rrule(::typeof(qrpos), A::U1Array)
    Q, R = qrpos(A)
    function back((dQ, dR))
        dA = copy(A)
        @assert Q.qn == dQ.qn
        # @assert R.qn == dR.qn
        Qqn, Qdir, Qdims, Qdiv = Q.qn, Q.dir, Q.dims, Q.division
        Rqn, Rdims = R.qn, R.dims
        Abdiv = blockdiv(A.dims)
        Qbdiv = blockdiv(Qdims)
        Qtensor = [reshape(@view(Q.tensor[Qbdiv[i]]), prod(Qdims[i][1:Qdiv]), prod(Qdims[i][Qdiv+1:end])) for i in 1:length(Qbdiv)]
        dQtensor = [reshape(@view(dQ.tensor[Qbdiv[i]]), prod(Qdims[i][1:Qdiv]), prod(Qdims[i][Qdiv+1:end])) for i in 1:length(Qbdiv)]

        Rbdiv = blockdiv(Rdims)
        Rtensor = [reshape(@view(R.tensor[Rbdiv[i]]), Rdims[i]...) for i in 1:length(Rbdiv)]
        if dR == ZeroTangent()
            dRtensor = ZeroTangent()
        else
            dRtensor = [reshape(@view(dR.tensor[Rbdiv[i]]), Rdims[i]...) for i in 1:length(Rbdiv)]
        end
        qs = A.ifZ2 ? map(x->sum(x[A.division+1:end]) % 2, A.qn) : map(x->sum(x[A.division+1:end] .* A.dir[A.division+1:end]), A.qn)
        for q in unique(qs)
            blockbackQR!(dA::U1Array, Abdiv, Qqn, Qdiv, Qdir, Qtensor, Rqn, Rtensor, dQtensor, dRtensor, q, A.ifZ2)
        end
        return NoTangent(), dA
    end
    return (Q, R), back
end

function blockbackQR!(dA::U1Array, Abdiv, Qqn, Qdiv, Qdir, Qtensor, Rqn, Rtensor, dQtensor, dRtensor, q, ifZ2)
    ind_A = ifZ2 ? findall(x->sum(x[Qdiv+1:end]) % 2 == q, Qqn) : findall(x->sum(x[Qdiv+1:end] .* Qdir[Qdiv+1:end]) == q, Qqn)
    m_j = unique(map(x->x[Qdiv+1:end], Qqn[ind_A]))
    m_i = unique(map(x->x[1:Qdiv], Qqn[ind_A]))

    ind = indexin([[i; m_j[1]] for i in m_i], Qqn)
    dQm = vcat(dQtensor[ind]...)
    Qm = vcat(Qtensor[ind]...)
    blockidims = [size(dQtensor[i],1) for i in ind]
    ind = indexin([[m_j[1]; m_j[1]]], Rqn)[1]
    dRm = dRtensor == ZeroTangent() ? ZeroTangent() : dRtensor[ind]
    Rm = Rtensor[ind]

    M = Array(Rm * dRm' - dQm' * Qm)
    dAm = (UpperTriangular(Rm + I * 1e-12) \ (dQm + Qm * _arraytype(Qm)(Hermitian(M, :L)))' )'

    for i in 1:length(m_i)
        ind = findfirst(x->x in [[m_i[i]; m_j[1]]], dA.qn)
        idim = sum(blockidims[1:i-1])+1:sum(blockidims[1:i])
        CUDA.@allowscalar dA.tensor[Abdiv[ind]] = vec(@view(dAm[idim, :]))
    end
end

function ChainRulesCore.rrule(::typeof(lqpos), A::U1Array)
    L, Q = lqpos(A)
    function back((dL, dQ))
        dA = copy(A)
        @assert Q.qn == dQ.qn
        # @assert L.qn == dL.qn
        Qqn, Qdir, Qdims, Qdiv = Q.qn, Q.dir, Q.dims, Q.division
        Lqn, Ldims = L.qn, L.dims
        Abdiv = blockdiv(A.dims)
        Qbdiv = blockdiv(Qdims)
        Qtensor = [reshape(@view(Q.tensor[Qbdiv[i]]), prod(Qdims[i][1:Qdiv]), prod(Qdims[i][Qdiv+1:end])) for i in 1:length(Qbdiv)]
        dQtensor = [reshape(@view(dQ.tensor[Qbdiv[i]]), prod(Qdims[i][1:Qdiv]), prod(Qdims[i][Qdiv+1:end])) for i in 1:length(Qbdiv)]

        Lbdiv = blockdiv(Ldims)
        Ltensor = [reshape(@view(L.tensor[Lbdiv[i]]), Ldims[i]...) for i in 1:length(Lbdiv)]
        if dL == ZeroTangent()
            dLtensor = ZeroTangent()
        else
            dLtensor = [reshape(@view(dL.tensor[Lbdiv[i]]), Ldims[i]...) for i in 1:length(Lbdiv)]
        end
        qs = A.ifZ2 ? map(x->x[1] % 2, A.qn) : map(x->x[1] * A.dir[1], A.qn)
        for q in unique(qs)
            blockbackLQ!(dA::U1Array, Abdiv, Qqn, Qdiv, Qdir, Qtensor, Lqn, Ltensor, dQtensor, dLtensor, q, A.ifZ2)
        end
        @show 333333333 typeof(dA)
        return NoTangent(), dA
    end
    return (L, Q), back
end

function blockbackLQ!(dA::U1Array, Abdiv, Qqn, Qdiv, Qdir, Qtensor, Lqn, Ltensor, dQtensor, dLtensor, q, ifZ2)
    ind_A = ifZ2 ? findall(x->x[1] % 2 == q, Qqn) : findall(x->x[1] * Qdir[1] == q, Qqn)
    m_j = unique(map(x->x[Qdiv+1:end], Qqn[ind_A]))
    m_i = unique(map(x->x[1], Qqn[ind_A]))

    ind = indexin([[m_i[1]; j] for j in m_j], Qqn)
    dQm = hcat(dQtensor[ind]...)
    Qm = hcat(Qtensor[ind]...)
    blockjdims = [size(dQtensor[i],2) for i in ind]
    ind = indexin([[m_i[1]; m_i[1]]], Lqn)[1]
    dLm = dLtensor == ZeroTangent() ? ZeroTangent() : dLtensor[ind]
    Lm = Ltensor[ind]
    
    M = Array(Lm' * dLm - dQm * Qm')
    dAm = LowerTriangular(Lm + I * 1e-12)' \ (dQm + _arraytype(Qm)(Hermitian(M, :L)) * Qm)

    for j in 1:length(m_j)
        ind = findfirst(x->x in [[m_i[1]; m_j[j]]], dA.qn)
        jdim = sum(blockjdims[1:j-1])+1:sum(blockjdims[1:j])
        CUDA.@allowscalar dA.tensor[Abdiv[ind]] = vec(@view(dAm[:, jdim]))
    end
end