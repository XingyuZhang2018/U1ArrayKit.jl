function add!!(y::DoubleArray, x::DoubleArray, α::Number, β::Number)
    add!!(y.real, x.real, α, β)
    add!!(y.imag, x.imag, α, β)
    return y
end

function inner(A::DoubleArray, B::DoubleArray) 
    return inner(A.real, B.real) - inner(A.imag, B.imag)
end

function scale!!(x::DoubleArray, α::Number)
    scale!!(x.real, α)
    scale!!(x.imag, α)
    return x
end

function scale!!(y::DoubleArray, x::DoubleArray, α::Number)
    scale!!(y.real, x.real, α)
    scale!!(y.imag, x.imag, α)
    return y
end

scale(A::DoubleArray, α::Number) = DoubleArray(scale(A.real, real(α)), scale(A.imag, real(α)))

zerovector(x::DoubleArray, ::Type{S}) where {S<:Number} = DoubleArray(zero(x.real), zero(x.imag))
