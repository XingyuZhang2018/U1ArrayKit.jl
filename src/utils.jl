#helper functions to handle array types
_mattype(::Array{T}) where {T} = Matrix
_mattype(::CuArray{T}) where {T} = CuMatrix
_mattype(::Adjoint{T, CuArray{T, 2 ,B}}) where {T,B} = CuMatrix
_mattype(::Symmetric{T, CuArray{T, 2, B}}) where {T,B} = CuMatrix

_arraytype(::Array) = Array
_arraytype(::CuArray) = CuArray
_arraytype(::Diagonal{T, Vector{T}}) where {T} = Array
_arraytype(::Diagonal{T, CuArray{T, 1, B}}) where {T, B} = CuArray
_arraytype(::U1Array) = U1Array
_arraytype(::Adjoint{T, Matrix{T}}) where {T} = Matrix
_arraytype(::Adjoint{T, CuArray{T, 2, B}}) where {T,B} = CuArray
_arraytype(::Transpose{T, Vector{T}}) where {T} = Vector
_arraytype(::Transpose{T, Matrix{T}}) where {T} = Matrix
_arraytype(::Transpose{T, CuArray{T, N, B}}) where {T,N,B} = CuArray