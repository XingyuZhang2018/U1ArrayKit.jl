function randU1double(atype, s...; kwarg...)
    indqn = [[0,1] for _ in 1:length(s)]
    indims = [getblockdims(DoublePEPSZ2(Int(sqrt(s))), s)[1] for s in s]
    double_real = randU1(atype, Float64, sum.(indims)...; indqn=indqn, indims=indims, ifZ2=true, f=[0], dir=[1 for _ in 1:length(s)], kwarg...)
    double_imag = randU1(atype, Float64, sum.(indims)...; indqn=indqn, indims=indims, ifZ2=true, f=[1], dir=[1 for _ in 1:length(s)])
    return DoubleArray(double_real, double_imag)
end

function IU1double(atype, D; kwarg...)
    indqn = [[0,1] for _ in 1:2]
    indims = getblockdims(DoublePEPSZ2(Int(sqrt(D))), D,D)
    double_real = IU1(atype, Float64, D; indqn=indqn, indims=indims, ifZ2=true, f=[0], dir=[-1,1])
    double_imag = nothing
    return DoubleArray(double_real, double_imag)
end