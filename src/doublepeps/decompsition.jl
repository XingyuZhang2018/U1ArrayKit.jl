function qrpos(A::DoubleArray) 
    # A = asComplexArray(A)
    # Q, R = qrpos(A)
    # Qdiv = blockdiv(Q.dims)
    # evenind = [sum(qn) % 2 == 0 for qn in Q.qn]
    # oddind = .!evenind
    # Q_real = U1Array(Q.qn[evenind], Q.dir, real(Q.tensor[vcat(Qdiv[evenind]...)]), Q.size, Q.dims[evenind], Q.division, true)
    # Q_imag = U1Array(Q.qn[oddind], Q.dir, imag(Q.tensor[vcat(Qdiv[oddind]...)]), Q.size, Q.dims[oddind], Q.division, true)
    # R_real = U1Array(R.qn, R.dir, real(R.tensor), R.size, R.dims, R.division, true)
    # R_imag = nothing
    s = Int.(sqrt.(size(A)))
    sitetypes = [DoublePEPSZ2(s) for s in s]
    dir = A.real.dir
    div = A.real.division
    A = reshape(asArray(sitetypes, asComplexArray(A)), prod(size(A)[1:div]), prod(size(A)[div+1:end]))
    @assert prod(size(A)[1:div]) > prod(size(A)[div+1:end]) "The number of rows must be greater than the number of columns"
    Q, R = qrpos!(A)
    Q = reshape(Q, s.^2...)
  
    s = Int.(sqrt.(size(Q)))
    sitetypes = [DoublePEPSZ2(s) for s in s]
    Q_real = asU1Array(sitetypes, real(Q); q=[0], dir=dir, division=div)
    Q_imag = asU1Array(sitetypes, imag(Q); q=[1], dir=dir, division=div)

    s = Int.(sqrt.(size(R)))
    sitetypes = [DoublePEPSZ2(s) for s in s]
    R_real = asU1Array(sitetypes, real(R); q=[0], dir=[-dir[end],dir[end]])
    R_imag = asU1Array(sitetypes, imag(R); q=[1], dir=[-dir[end],dir[end]])

    return DoubleArray(Q_real, Q_imag), DoubleArray(R_real, R_imag)
end 

function lqpos(A::DoubleArray)
    s = Int.(sqrt.(size(A)))
    sitetypes = [DoublePEPSZ2(s) for s in s]
    dir = A.real.dir
    div = A.real.division
    A = reshape(asArray(sitetypes, asComplexArray(A)), prod(size(A)[1:div]), prod(size(A)[div+1:end]))
    @assert prod(size(A)[1:div]) < prod(size(A)[div+1:end]) "The number of rows must be less than the number of columns"
    L, Q = lqpos!(A)
    Q = reshape(Q, s.^2...)
  
    s = Int.(sqrt.(size(L)))
    sitetypes = [DoublePEPSZ2(s) for s in s]
    L_real = asU1Array(sitetypes, real(L); q=[0], dir=[dir[1],-dir[1]])
    L_imag = asU1Array(sitetypes, imag(L); q=[1], dir=[dir[1],-dir[1]])

    s = Int.(sqrt.(size(Q)))
    sitetypes = [DoublePEPSZ2(s) for s in s]
    Q_real = asU1Array(sitetypes, real(Q); q=[0], dir=dir)
    Q_imag = asU1Array(sitetypes, imag(Q); q=[1], dir=dir)
    return DoubleArray(L_real, L_imag), DoubleArray(Q_real, Q_imag)
end
