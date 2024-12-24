struct DoubleArray{T, N} <: AbstractArray{T, N}
    real::AbstractArray
    imag::AbstractArray
    function DoubleArray(real::AbstractArray{T, N}, imag::AbstractArray{T, N}) where {T, N}
        if size(real) != size(imag)
            throw(ArgumentError("size(real) != size(imag)"))
        end
        new{T, N}(real, imag)
    end
    function DoubleArray(A::AbstractArray{T, N}) where {T, N}
        new{T, N}(real(A), imag(A))
    end
end

*(x::DoubleArray, y::DoubleArray) = DoubleArray(x.real * y.real - x.imag * y.imag, x.real * y.imag + x.imag * y.real)

size(A::DoubleArray) = size(A.real)
function show(::IOBuffer, A::DoubleArray)
    println("real: \n", A.real)
    println("imag: \n", A.imag)
end

reshape(A::DoubleArray, s::Vararg{Int}) = DoubleArray(reshape(A.real, s), reshape(A.imag, s))
tensorpermute(A::DoubleArray, perm) = DoubleArray(tensorpermute(A.real, perm), tensorpermute(A.imag, perm))