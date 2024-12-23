
"""
    div = division(a::NTuple{Na, Int}, b::NTuple{Nb, Int}) where {Na, Nb}

give the reshape division of b by a, where b is the original shape and a is the new shape.
"""
function division(a::NTuple{Na, Int}, b::NTuple{Nb, Int}) where {Na, Nb}
    prod(a) != prod(b) && throw(Base.error("$a and $b must have the same product"))
    Na > Nb && throw(Base.error("$a must be shorter than $b"))
    div = Int[zeros(Int, Na)..., Nb]
    for i in 2:Na
        idiv = div[i-1] + 1
        p = b[idiv]
        while p != a[i-1]
            idiv += 1 
            p *= b[idiv]
        end
        div[i] = idiv
    end
    [div[i] + 1 : div[i+1] for i in 1:Na]
end

"""
    U1reshape(A::U1Array{T, N}, a::Int...) where {T, N}

U1reshape only for shape `(D,D,D,D,D,D,D,D) <-> (D^2,D^2,D^2,D^2)` and `(χ,D,D,χ) <-> (χ,D^2,χ)`, and the high-oreder U1tensor is from randU1 or zerosU1 function.
"""
U1reshape(A::U1Array, s::Tuple{Vararg{Int}}; kwarg...) = U1reshape(A, s...; kwarg...)
function U1reshape(A::U1Array{T, N}, s::Int...; reinfo = nothing) where {T <: Number, N}
    dims = A.dims
    bdiv = blockdiv(dims) 
    tensor = [reshape(@view(A.tensor[bdiv[i]]), dims[i]...) for i in 1:length(bdiv)]
    A = U1Array(A.qn, A.dir, tensor, A.size, dims, A.division, A.ifZ2)
    U1reshape(A, s; reinfo = reinfo)
end

function U1reshape(A::U1Array{T, N}, s::Int...; reinfo) where {T <: AbstractArray, N}
    atype = typeof(A.tensor[1]) <: CuArray ? CuArray : Array
    etype = eltype(A.tensor[1])
    if N > length(s)
        _, _, _, indqn, indims, _, _ = reinfo
        cA = zerosU1(Array, ComplexF64, size(A)...; dir=A.dir, indqn=indqn, indims=indims, ifZ2=A.ifZ2)
        qndiff = setdiff(cA.qn, A.qn)
        supind = indexin(qndiff, cA.qn)
        Aqn = [A.qn; cA.qn[supind]]
        cAbdiv = blockdiv(cA.dims)
        Atensor = [A.tensor; [reshape(cA.tensor[cAbdiv[supind[i]]], cA.dims[supind[i]]...) for i in 1:length(supind)]]
        exchangeind = indexin(cA.qn, Aqn)
        Aqn = cA.qn
        Adims = cA.dims
        Atensor = Atensor[exchangeind]
        div = division(s, size(A))
        reqn = A.ifZ2 ? [[sum(p[d]) % 2 for d in div] for p in Aqn] : [[sum(p[d] .* A.dir[d]) for d in div] for p in Aqn]
        redims = [[prod(dims[d]) for d in div] for dims in Adims]
        retensor = [reshape(t, s...) for (t, s) in zip(map(Array, Atensor), redims)]
        ureqn = unique(reqn)
        retensors = Vector{atype{etype}}()
        choosesilces = [[] for _ in 1:length(ureqn)]
        chooseinds = [[] for _ in 1:length(ureqn)]
        for i in 1:length(ureqn)
            q = ureqn[i]
            bulkind = findall(x->x in [q], reqn)
            oriqn = Aqn[bulkind]
        
            indqnfrom = [unique(map(x->x[div], oriqn)) for div in div]
            rebulkdims = [[prod(map((x,y,z)->x[indexin(y, z)...], indims[div[i]], indqnfrom, indqn[div[i]])) for indqnfrom in indqnfrom[i]] for i in 1:length(indqnfrom)]
            # @show indqnfrom
            # indqnfrom = [[[0, 0], [1, -1]], [[0, 0], [1, -1]], [[0, 0], [1, -1]], [[0, 0], [1, -1]]]
            # rebulkdims = [[1, 4], [1, 4], [1, 4], [1, 4]]
            silce = [[(sum(rebulkdims[1:(i-1)]) + 1) : sum(rebulkdims[1:i]) for i in 1:length(rebulkdims)] for rebulkdims in rebulkdims]
            tensor = atype(zeros(etype, map(sum, rebulkdims)...))
            for j in 1:length(bulkind)
                chooseind = [indexin([oriqn[j][div[i]]], indqnfrom[i]) for i in 1:length(div)]
                choosesilce = map((s,i)->s[i...], silce, chooseind)
                tensor[choosesilce...] = retensor[bulkind[j]]
                push!(choosesilces[i], choosesilce)
                push!(chooseinds[i], bulkind[j])
            end
            push!(retensors, tensor)
        end
        nozeroind = norm.(retensors) .!= 0
        dims = map(x -> collect(size(x)), retensors)[nozeroind]
        dir = [A.dir[d][end] for d in div]     # last dir of reshape
        qn = A.ifZ2 ? ureqn[nozeroind] : map(qn->qn .* dir, ureqn)[nozeroind]
        p = sortperm(qn)
        tensor = atype(vcat(map(vec, retensors[nozeroind][p])...))
        U1Array(qn[p], dir, tensor, s, dims[p], 1, A.ifZ2), (choosesilces, chooseinds, A.dir, indqn, indims, Aqn, Adims)
    else
        choosesilces, chooseinds, redir, indqn, indims, reqn, redims = reinfo
        retensors = Array{Array,1}(undef, sum(length.(chooseinds)))
        div = division(size(A), s)
        ureqn = A.ifZ2 ? unique([[sum(p[d]) % 2 for d in div] for p in reqn]) : unique([[sum(p[d] .* redir[d]) for d in div] for p in reqn])
        exchangeind = A.ifZ2 ? indexin(ureqn, A.qn) : indexin(ureqn, map(qn->qn .* A.dir, A.qn))
        # @show ureqn A.qn exchangeind
        Atensor = A.tensor[exchangeind]
        for i in 1:length(choosesilces)
            for j in 1:length(choosesilces[i])
                retensors[chooseinds[i][j]] = reshape(Array(Atensor[i][choosesilces[i][j]...]), redims[chooseinds[i][j]]...)
            end
        end
        nozeroind = norm.(retensors) .!= 0
        dims = map(x -> collect(size(x)), retensors)[nozeroind]
        qn = reqn[nozeroind]
        p = sortperm(qn)
        tensor = atype(vcat(map(vec, retensors[nozeroind][p])...))
        U1Array(qn[p], redir, tensor, s, dims[p], 1, A.ifZ2), (choosesilces, chooseinds, redir, indqn, indims, reqn, redims)
    end
end

function U1reshapeinfo(s, sizeA, dir, indqn, indims, ifZ2)
    length(sizeA) < length(s) && throw(Base.error("$sizeA must be longer than $s"))
    div = division(s, sizeA)
    A = zerosU1(Array, ComplexF64, sizeA...; dir=dir, indqn=indqn, indims=indims, ifZ2=ifZ2)
    reqn = ifZ2 ? [[sum(p[d]) % 2 for d in div] for p in A.qn] : [[sum(p[d] .* dir[d]) for d in div] for p in A.qn]
    ureqn = unique(reqn)
    choosesilces = [[] for _ in 1:length(ureqn)]
    chooseinds = [[] for _ in 1:length(ureqn)]
    Aqn = A.qn
    for i in 1:length(ureqn)
        q = ureqn[i]
        bulkind = findall(x->x in [q], reqn)
        oriqn = Aqn[bulkind]
    
        indqnfrom = [unique(map(x->x[div], oriqn)) for div in div]
        rebulkdims = [[prod(map((x,y,z)->x[indexin(y, z)...], indims[div[i]], indqnfrom, indqn[div[i]])) for indqnfrom in indqnfrom[i]] for i in 1:length(indqnfrom)]
        # @show indqnfrom
        # indqnfrom = [[[0, 0], [1, -1]], [[0, 0], [1, -1]], [[0, 0], [1, -1]], [[0, 0], [1, -1]]]
        # rebulkdims = [[1, 4], [1, 4], [1, 4], [1, 4]]
        silce = [[(sum(rebulkdims[1:(i-1)]) + 1) : sum(rebulkdims[1:i]) for i in 1:length(rebulkdims)] for rebulkdims in rebulkdims]
        for j in 1:length(bulkind)
            chooseind = [indexin([oriqn[j][div[i]]], indqnfrom[i]) for i in 1:length(div)]
            choosesilce = map((s,i)->s[i...], silce, chooseind)
            push!(choosesilces[i], choosesilce)
            push!(chooseinds[i], bulkind[j])
        end
    end
    choosesilces, chooseinds, dir, indqn, indims, Aqn, A.dims
end