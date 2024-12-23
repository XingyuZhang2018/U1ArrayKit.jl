randU1(sitetype::AbstractSiteType, atype, dtype, s...; dir::Vector{Int}, f::Vector{Int}=[0]) = randU1(atype, dtype, s...; dir=dir, indqn=getqrange(sitetype, s), indims=getblockdims(sitetype, s), f=f, ifZ2=sitetype.ifZ2)

function randU1(atype, dtype, s...; dir::Vector{Int}, indqn::Vector{Vector{Int}}, indims::Vector{Vector{Int}}, f::Vector{Int}=[0], ifZ2::Bool)
    s != Tuple(map(sum, indims)) && throw(Base.error("$s is not valid"))
    L = length(dir)
    qn = Vector{Vector{Int}}()
    tensor = Vector{atype{dtype}}()
    dims = Vector{Vector{Int}}()
    @inbounds for i in CartesianIndices(Tuple(length.(indqn)))
        qni = [indqn[j][i.I[j]] for j in 1:L]
        qnisum = ifZ2 ? sum(qni) % 2 : sum(qni .* dir)
        if qnisum in f
            bulkdims = [indims[j][i.I[j]] for j in 1:L]
            push!(qn, qni)
            push!(dims, bulkdims)
            push!(tensor, atype(rand(dtype, bulkdims...)))
        end
    end
    p = sortperm(qn)
    tensor = vcat(map(vec, tensor[p])...)
    U1Array(qn[p], dir, tensor, s, dims[p], 1, ifZ2)
end

zerosU1(sitetype::AbstractSiteType, atype, dtype, s...; dir::Vector{Int}, f::Vector{Int}=[0]) = zerosU1(atype, dtype, s...; dir=dir, indqn=getqrange(sitetype, s), indims=getblockdims(sitetype, s), f=f, ifZ2=sitetype.ifZ2)

function zerosU1(atype, dtype, s...; dir::Vector{Int}, indqn::Vector{Vector{Int}}, indims::Vector{Vector{Int}}, f::Vector{Int}=[0], ifZ2::Bool)
    s != Tuple(map(sum, indims)) && throw(Base.error("$s is not valid"))
    L = length(dir)
    qn = Vector{Vector{Int}}()
    tensor = Vector{atype{dtype}}()
    dims = Vector{Vector{Int}}()
    @inbounds for i in CartesianIndices(Tuple(length.(indqn)))
        qni = [indqn[j][i.I[j]] for j in 1:L]
        qnisum = ifZ2 ? sum(qni) % 2 : sum(qni .* dir)
        if qnisum in f
            bulkdims = [indims[j][i.I[j]] for j in 1:L]
            push!(qn, qni)
            push!(dims, bulkdims)
            push!(tensor, atype(zeros(dtype, bulkdims...)))
        end
    end
    p = sortperm(qn)
    tensor = vcat(map(vec, tensor[p])...)
    U1Array(qn[p], dir, tensor, s, dims[p], 1, ifZ2)
end

zero(A::U1Array) = U1Array(A.qn, A.dir, zero(A.tensor), A.size, A.dims, A.division, A.ifZ2)

IU1(sitetype::AbstractSiteType, atype, dtype, D; dir::Vector{Int}, f::Vector{Int}=[0]) = IU1(atype, dtype, D; dir=dir, indqn=getqrange(sitetype, D,D), indims=getblockdims(sitetype, D,D), f=f, ifZ2=sitetype.ifZ2)
function IU1(atype, dtype, D; dir::Vector{Int}, indqn::Vector{Vector{Int}}, indims::Vector{Vector{Int}}, f::Vector{Int}=[0], ifZ2::Bool)
    (D, D) != Tuple(map(sum, indims)) && throw(Base.error("$D is not valid"))
    L = length(dir)
    qn = Vector{Vector{Int}}()
    tensor = Vector{atype{dtype}}()
    dims = Vector{Vector{Int}}()
    @inbounds for i in CartesianIndices(Tuple(length.(indqn)))
        qni = [indqn[j][i.I[j]] for j in 1:L]
        qnisum = ifZ2 ? sum(qni) % 2 : sum(qni .* dir)
        if qnisum in f
            bulkdims = [indims[j][i.I[j]] for j in 1:L]
            push!(qn, qni)
            push!(dims, bulkdims)
            push!(tensor, atype{dtype}(I, bulkdims...))
        end
    end
    p = sortperm(qn)
    tensor = vcat(map(vec, tensor[p])...)
    U1Array(qn[p], dir, tensor, (D, D), dims[p], 1, ifZ2)
end

randU1DiagMatrix(sitetype::AbstractSiteType, atype, dtype, D; dir::Vector{Int}, f::Vector{Int}=[0]) = randU1DiagMatrix(atype, dtype, D; dir=dir, indqn=getqrange(sitetype, D,D), indims=getblockdims(sitetype, D,D), f=f, ifZ2=sitetype.ifZ2)
function randU1DiagMatrix(atype, dtype, D; dir::Vector{Int}, indqn::Vector{Vector{Int}}, indims::Vector{Vector{Int}}, f::Vector{Int}=[0], ifZ2::Bool)
    (D, D) != Tuple(map(sum, indims)) && throw(Base.error("$D is not valid"))
    L = length(dir)
    qn = Vector{Vector{Int}}()
    tensor = Vector{atype{dtype}}()
    dims = Vector{Vector{Int}}()
    @inbounds for i in CartesianIndices(Tuple(length.(indqn)))
        qni = [indqn[j][i.I[j]] for j in 1:L]
        qnisum = ifZ2 ? sum(qni) % 2 : sum(qni .* dir)
        if qnisum in f
            bulkdims = [indims[j][i.I[j]] for j in 1:L]
            push!(qn, qni)
            push!(dims, bulkdims)
            push!(tensor, atype(diagm(rand(dtype, bulkdims[1]))))
        end
    end
    p = sortperm(qn)
    tensor = vcat(map(vec, tensor[p])...)
    U1Array(qn[p], dir, tensor, (D, D), dims[p], 1, ifZ2)
end