struct DoubleArray{T, N} <: AbstractArray{T, N}
    real
    imag
    function DoubleArray(real::AbstractArray{T, N}, imag::AbstractArray{T, N}) where {T, N}
        if size(real) != size(imag)
            throw(ArgumentError("size(real) != size(imag)"))
        end
        new{T, N}(real, imag)
    end
    function DoubleArray(A::AbstractArray{T, N}) where {T, N}
        new{T, N}(real(A), imag(A))
    end
    function DoubleArray(real::AbstractArray{T, N}, imag::Nothing) where {T, N}
        new{T, N}(real, imag)
    end
    function DoubleArray(real::Nothing, imag::AbstractArray{T, N}) where {T, N}
        new{T, N}(real, imag)
    end
end

function *(x::DoubleArray, y::DoubleArray)
    if y.imag isa Nothing
        return DoubleArray(x.real * y.real, x.imag * y.real)
    else
        return DoubleArray(x.real * y.real - x.imag * y.imag, x.real * y.imag + x.imag * y.real)
    end
end

-(A::DoubleArray) = DoubleArray(-A.real, -A.imag)
-(A::DoubleArray, B::DoubleArray) = DoubleArray(A.real - B.real, A.imag - B.imag)
+(A::DoubleArray, B::DoubleArray) = DoubleArray(A.real + B.real, A.imag + B.imag)
adjoint(A::DoubleArray) = DoubleArray(adjoint(A.real), -adjoint(A.imag))
conj(A::DoubleArray) = DoubleArray(A.real, -A.imag)
norm(A::DoubleArray) = norm(asComplexArray(A))

size(A::DoubleArray) = size(A.real)
function show(::IOBuffer, A::DoubleArray)
    @show A.real A.imag
end

≈(A::DoubleArray, B::DoubleArray) = (A.real ≈ B.real && A.imag ≈ B.imag)

reshape(A::DoubleArray, s::Vararg{Int}) = DoubleArray(reshape(A.real, s), reshape(A.imag, s))
tensorpermute(A::DoubleArray, perm) = DoubleArray(tensorpermute(A.real, perm), tensorpermute(A.imag, perm))