struct DoubleArray{T, N} <: AbstractArray{T, N}
    real
    imag
    function DoubleArray(real::AbstractArray{T, N}, imag::AbstractArray{T, N}) where {T<:Real, N}
        if size(real) != size(imag)
            throw(ArgumentError("size(real) != size(imag)"))
        end
        new{T, N}(real, imag)
    end
    function DoubleArray(A::AbstractArray{T, N}) where {T, N}
        new{T, N}(real(A), imag(A))
    end
    function DoubleArray(real::AbstractArray{T, N}, imag::Nothing) where {T<:Real, N}
        new{T, N}(real, imag)
    end
    function DoubleArray(real::Nothing, imag::AbstractArray{T, N}) where {T<:Real, N}
        new{T, N}(real, imag)
    end
end

function *(x::DoubleArray, y::DoubleArray)
    if y.imag isa Nothing && x.imag isa Nothing
        return DoubleArray(x.real * y.real, nothing)
    elseif y.imag isa Nothing
        return DoubleArray(x.real * y.real, x.imag * y.real)
    elseif x.imag isa Nothing
        return DoubleArray(x.real * y.real, x.real * y.imag)
    else
        return DoubleArray(x.real * y.real - x.imag * y.imag, x.real * y.imag + x.imag * y.real)
    end
end

rmul!(x::DoubleArray, y::Number) = (rmul!(x.real, y); rmul!(x.imag, y); x)

-(A::DoubleArray) = DoubleArray(-A.real, -A.imag)
-(A::DoubleArray, B::DoubleArray) = A.imag isa Nothing ? DoubleArray(A.real - B.real, -B.imag) : DoubleArray(A.real - B.real, A.imag - B.imag)
+(A::DoubleArray, B::DoubleArray) = DoubleArray(A.real + B.real, A.imag + B.imag)
*(A::DoubleArray, B::Number) = DoubleArray(A.real * B, A.imag * B)
*(B::Number, A::DoubleArray) = DoubleArray(A.real * B, A.imag * B)
/(A::DoubleArray, B::Number) = DoubleArray(A.real / B, A.imag / B)
adjoint(A::DoubleArray) = DoubleArray(adjoint(A.real), -adjoint(A.imag))
conj(A::DoubleArray) = DoubleArray(A.real, -A.imag)
norm(A::DoubleArray) = norm(asComplexArray(A))
norm(A::DoubleArray, p::Real) = norm(asComplexArray(A), p)
normalize!(A::DoubleArray) = (n = norm(A); rmul!(A, 1/n))
normalize!(A::DoubleArray, p::Real) = (n = norm(A, p); rmul!(A, 1/n))
similar(A::DoubleArray) = DoubleArray(similar(A.real), similar(A.imag))
copy(A::DoubleArray) = DoubleArray(copy(A.real), copy(A.imag))
size(A::DoubleArray) = size(A.real)
function show(::IOBuffer, A::DoubleArray)
    @show A.real A.imag
end

≈(A::DoubleArray, B::DoubleArray) = (A.real ≈ B.real && A.imag ≈ B.imag)
Array(A::DoubleArray) = DoubleArray(Array(A.real), Array(A.imag))
CuArray(A::DoubleArray) = DoubleArray(CuArray(A.real), CuArray(A.imag))

reshape(A::DoubleArray, s::Tuple{Vararg{Int}}) = reshape(A, s...)
reshape(A::DoubleArray, s::Vararg{Int}) = A.imag isa Nothing ? DoubleArray(reshape(A.real, s), A.imag) : DoubleArray(reshape(A.real, s), reshape(A.imag, s))
permutedims(A::DoubleArray, perm) = tensorpermute(A, perm)
tensorpermute(A::DoubleArray, perm) = A.imag isa Nothing ? DoubleArray(tensorpermute(A.real, perm), A.imag) : DoubleArray(tensorpermute(A.real, perm), tensorpermute(A.imag, perm))

