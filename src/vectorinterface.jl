function add!!(y::U1Array, x::U1Array, α::Number, β::Number)
    add!!(y.tensor, x.tensor, α, β)
    return y
end

function inner(A::U1Array, B::U1Array) 
    @assert A.qn == B.qn
    return inner(A.tensor, B.tensor)
end

function scale!!(x::U1Array, α::Number)
    scale!!(x.tensor, α)
    return x
end

function scale!!(y::U1Array, x::U1Array, α::Number)
    scale!!(y.tensor, x.tensor, α)
    return y
end

scale(A::U1Array, α::Number) = U1Array(A.qn, A.dir, scale(A.tensor, α) , A.size, A.dims, A.division, A.ifZ2)

zerovector(x::U1Array, ::Type{S}) where {S<:Number} = zero(x)
