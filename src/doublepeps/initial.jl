function randU1double(atype, s...; kwarg...)
    indqn = [[0,1] for _ in 1:length(s)]
    indims = [getblockdims(DoublePEPSZ2(s), s^2)[1] for s in s]
    double_real = randU1(atype, Float64, sum.(indims)...; indqn=indqn, indims=indims, ifZ2=true, f=[0], kwarg...)
    double_imag = randU1(atype, Float64, sum.(indims)...; indqn=indqn, indims=indims, ifZ2=true, f=[1], kwarg...)
    return DoubleArray(double_real, double_imag)
end

function IU1double(atype, D; kwarg...)
    indqn = [[0,1] for _ in 1:2]
    indims = getblockdims(DoublePEPSZ2(D), D^2,D^2)
    double_real = IU1(atype, Float64, D^2; indqn=indqn, indims=indims, ifZ2=true, f=[0], kwarg...)
    double_imag = zerosU1(atype, Float64, D^2, D^2; indqn=indqn, indims=indims, ifZ2=true, f=[1], kwarg...)
    return DoubleArray(double_real, double_imag)
end