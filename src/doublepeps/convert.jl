asComplexArray(A::DoubleArray) = A.real + 1im * A.imag
asArray(stype, A::DoubleArray) = DoubleArray(asArray(stype, A.real), asArray(stype, A.imag))

function convert_bilayer_Z2_front(M::AbstractArray{T, N}) where {T, N}
    D = Int(sqrt(size(M,1)))
    M′ = zeros(D,D,prod(size(M))÷D^2)
    M = reshape(M,D,D,prod(size(M))÷D^2)
    for i in 1:D, j in 1:D, k in 1:prod(size(M))÷D^2
        if i == j
            M′[i,j,k] = M[i,j,k]
        elseif i < j
            M′[i,j,k] = (M[i,j,k] + M[j,i,k])/sqrt(2)
        else
            M′[i,j,k] = (M[i,j,k] - M[j,i,k])/sqrt(2) 
        end
    end

    return reshape(M′,[D^2 for i in 1:N]...)
end

function convert_bilayer_Z2(M::AbstractArray{T, N}) where {T, N}
    for i in 1:N
        i != 1 && (M = permutedims(M, [2, [3:N; 1]...]))
        M = convert_bilayer_Z2_front(M)
    end
    return permutedims(M, [2, [1; 3:N]...])
end

function convert_bilayer_Z2(M::DoubleArray)
    DoubleArray(convert_bilayer_Z2(M.real), convert_bilayer_Z2(M.imag))
end

function asU1Array(sitetype, M::DoubleArray; kwarg...)
    return DoubleArray(asU1Array(sitetype, M.real; q=[0], kwarg...), asU1Array(sitetype, M.imag; q=[1], kwarg...))
end