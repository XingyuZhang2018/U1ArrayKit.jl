"""
    blockdiv(dims) 

get divsions of block U1 array from U1 contious storge block dims 
"""
function blockdiv(dims::Vector{Vector{Int}}) 
    blocklen = map(prod, dims)
    [sum(blocklen[1 : i - 1]) + 1 : sum(blocklen[1 : i]) for i in 1:length(blocklen)]
end

function qndims(A::U1Array, ind::Int)     
    indqn = Int64[]
    indims = Int64[]
    qn = A.qn 
    dims = A.dims 
    for i in 1:length(A.qn)
        if !(qn[i][ind] in indqn)
            push!(indqn, qn[i][ind]) 
            push!(indims, dims[i][ind]) 
        end 
    end   
    return indqn, indims
end

getq(sitetype, s::Int...) = map(s -> [indextoqn(sitetype, i) for i = 1:s], s)
getqrange(sitetype, s::Tuple{Vararg{Int}}) = getqrange(sitetype, s...)
getqrange(sitetype, s::Int...) = (q = getq(sitetype, s...); [map(q -> sort(unique(q)), q)...])
getshift(qrange) = map(q -> abs(q[1]), qrange) .+ 1

"""
    bkdims = getblockdims(size::Int...)

distribute dims of different part dims of U1 tensor bulk by bits division
"""
getblockdims(sitetype, s::Tuple{Vararg{Int}}) = getblockdims(sitetype, s...)
function getblockdims(sitetype, s::Int...)
    q = getq(sitetype, s...)
    [map(q -> [sum(q .== i) for i in sort(unique(q))], q)...]
end

function U1selection(sitetype, indqn::Vector{Int}, indims::Vector{Int})
    maxs = sum(indims)
    q = [indextoqn(sitetype, i) for i = 1:maxs]
    [q .== i for i in sort(unique(q))]
end

function U1selection(sitetype, maxs::Int)
    q = [indextoqn(sitetype, i) for i = 1:maxs]
    [q .== i for i in sort(unique(q))]
end

function asArray(sitetype, A::U1Array{T,N}; indqn::Vector{Vector{Int}}=getqrange(sitetype, size(A)), indims::Vector{Vector{Int}}=getblockdims(sitetype, size(A))) where {T <: Number, N}
    atype = _arraytype(A.tensor)
    tensor = zeros(T, size(A))
    Aqn = A.qn
    Atensor = A.tensor
    qlist = [U1selection(sitetype, indqn[i], indims[i]) for i = 1:N]
    div = blockdiv(A.dims)
    for i in 1:length(Aqn)
        tensor[[qlist[j][indexin([Aqn[i][j]], indqn[j])...] for j = 1:N]...] = Array(Atensor[div[i]])
    end
    atype(tensor)
end

function asArray(sitetype, A::U1Array{T,N}; indqn::Vector{Vector{Int}}=getqrange(sitetype, size(A)), indims::Vector{Vector{Int}}=getblockdims(sitetype, size(A))) where {T <: AbstractArray, N}
    atype = _arraytype(A.tensor[1])
    etype = eltype(A.tensor[1])
    tensor = zeros(etype, size(A))
    Aqn = A.qn
    Atensor = A.tensor
    qlist = [U1selection(sitetype, indqn[i], indims[i]) for i = 1:N]
    for i in 1:length(Aqn)
        tensor[[qlist[j][indexin([Aqn[i][j]], indqn[j])...] for j = 1:N]...] = Array(Atensor[i])
    end
    atype(tensor)
end

"""
    qn = getqn(dir::Vector{Int}, indqn; q::Vector{Int}=[0])

give the qn of length L
"""
function getqn(dir::Vector{Int}, indqn::Vector{Vector{Int}}; q::Vector{Int}=[0], ifZ2)
    L = length(dir)
    qn = Vector{Vector{Int}}()
    @inbounds for i in CartesianIndices(Tuple(length.(indqn)))
        qni = [indqn[j][i.I[j]] for j in 1:L]
        qnisum = ifZ2 ? sum(qni) % 2 : sum(qni .* dir)
        if qnisum in q
            push!(qn, qni)
        end
    end
    sort!(qn)
end

function deletezeroblock(A::U1Array)
    Adims = A.dims
    Abdiv = blockdiv(Adims)
    Atensor = [A.tensor[Abdiv[i]] for i in 1:length(Abdiv)]
    nozeroind = norm.(Atensor) .!== 0
    Atensor = vcat(Atensor[nozeroind]...)
    U1Array(A.qn[nozeroind], A.dir, A.tensor, A.size, A.dims[nozeroind], A.division, A.ifZ2)
end

# have Bugs with CUDA@v3.5.0, rely on https://github.com/JuliaGPU/CUDA.jl/issues/1304
# which is fixed in new vervion, but its allocation is abnormal
function asU1Array(sitetype, A::AbstractArray{T,N}; dir::Vector{Int}, indqn::Vector{Vector{Int}}=getqrange(sitetype, size(A)), indims::Vector{Vector{Int}}=getblockdims(sitetype, size(A)), q::Vector{Int}=[0]) where {T, N}
    atype = _arraytype(A)
    Aarray = Array(A)
    qlist = [U1selection(sitetype, indqn[i], indims[i]) for i = 1:N]
    Aqn = getqn(dir, indqn; q = q, ifZ2=sitetype.ifZ2)
    tensor = [atype(Aarray[[qlist[j][indexin([Aqn[i][j]], indqn[j])...] for j = 1:N]...]) for i in 1:length(Aqn)]
    dims = map(x -> collect(size(x)), tensor)
    nozeroind = norm.(tensor) .!= 0
    tensor = atype{T}(vcat(map(vec, tensor[nozeroind])...))
    U1Array(Aqn[nozeroind], dir, tensor, size(A), dims[nozeroind], 1, sitetype.ifZ2)
end