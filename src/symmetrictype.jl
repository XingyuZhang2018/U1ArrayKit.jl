@with_kw struct SymmetricType
    symmetry = :U1
    stype::AbstractSiteType
    atype = Array
    dtype = ComplexF64
end

getsymmetry(::AbstractArray) = :none
getsymmetry(::U1Array) = :U1

getdir(::AbstractArray) = nothing

randinitial(ST::SymmetricType, size...; kwarg...) = randinitial(Val(ST.symmetry), ST.stype, ST.atype, ST.dtype, size...; kwarg...)
randinitial(::Val{:none}, atype, dtype, size...; kwarg...) = atype(rand(dtype, size...))
randinitial(::Val{:none}, sitetype::AbstractSiteType, atype, dtype, size...; kwarg...) = atype(rand(dtype, size...))
randinitial(::Val{:U1}, atype, dtype, size...; kwarg...) = randU1(atype, dtype, size...; kwarg...)
randinitial(::Val{:U1}, sitetype::AbstractSiteType, atype, dtype, size...; kwarg...) = randU1(sitetype,atype, dtype, size...; kwarg...)

function randinitial(A::AbstractArray{T, N}, size...; kwarg...) where {T, N}
    atype = typeof(A) <: Union{Array, CuArray} ? _arraytype(A) : _arraytype(A.tensor)
    randinitial(Val(getsymmetry(A)), atype, T, size...; ifZ2=A.ifZ2, kwarg...)
end

Iinitial(ST::SymmetricType, size...; kwarg...) = Iinitial(Val(ST.symmetry), ST.stype, ST.atype, ST.dtype, size...; kwarg...)
Iinitial(::Val{:none}, atype, dtype, D; kwarg...) = atype{dtype}(I, D, D)
Iinitial(::Val{:none}, sitetype::AbstractSiteType, atype, dtype, D; kwarg...) = atype{dtype}(I, D, D)
Iinitial(::Val{:U1}, atype, dtype, D; kwarg...) = IU1(atype, dtype, D; kwarg...)
Iinitial(::Val{:U1}, sitetype::AbstractSiteType, atype, dtype, D; kwarg...) = IU1(sitetype, atype, dtype, D; kwarg...)

function Iinitial(A::AbstractArray{T, N}, D; kwarg...) where {T, N}
    atype = typeof(A) <: Union{Array, CuArray} ? _arraytype(A) : _arraytype(A.tensor)
    Iinitial(Val(getsymmetry(A)), atype, ComplexF64, D; ifZ2=A.ifZ2, kwarg...)
end

zerosinitial(ST::SymmetricType, size...; kwarg...) = zerosinitial(Val(ST.symmetry), ST.stype, ST.atype, ST.dtype, size...; kwarg...)
zerosinitial(::Val{:none}, atype, dtype, size...; kwarg...) = atype(zeros(dtype, size...))
zerosinitial(::Val{:none}, sitetype::AbstractSiteType, atype, dtype, size...; kwarg...) = atype(zeros(dtype, size...))
zerosinitial(::Val{:U1}, atype, dtype, size...; kwarg...) = zerosU1(atype, dtype, size...; kwarg...)
zerosinitial(::Val{:U1}, sitetype::AbstractSiteType, atype, dtype, size...; kwarg...) = zerosU1(sitetype, atype, dtype, size...; kwarg...)

function zerosinitial(A::AbstractArray{T, N}, size...; kwarg...) where {T, N}
    atype = typeof(A) <: Union{Array, CuArray} ? _arraytype(A) : _arraytype(A.tensor)
    zerosinitial(Val(getsymmetry(A)), atype, T, size...; ifZ2=A.ifZ2, kwarg...)
end

asArray(A::Union{Array, CuArray}) = A

"""
    asSymmetryArray(A::AbstractArray, symmetry; dir = nothing)

Transform Array to a SymmetryArray.
now supports:
    `:none`
    `:Z2`
    `:U1`
"""
asSymmetryArray(A::AbstractArray, ST; kwarg...) = asSymmetryArray(A, Val(ST.symmetry), ST.stype; kwarg...)
asSymmetryArray(A::AbstractArray, ::Val{:none}; kwarg...) = A
asSymmetryArray(A::AbstractArray, ::Val{:none}, sitetype::AbstractSiteType; kwarg...) = A
asSymmetryArray(A::AbstractArray, ::Val{:U1}, sitetype::AbstractSiteType;  kwarg...) = asU1Array(sitetype, A; kwarg...)

symmetryreshape(A::AbstractArray, s...; kwarg...) = reshape(A, s...), nothing
symmetryreshape(A::U1Array, s...; kwarg...) = U1reshape(A, s...; kwarg...)