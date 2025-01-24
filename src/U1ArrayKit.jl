module U1ArrayKit

using CUDA
using ChainRulesCore
using LinearAlgebra
using OMEinsum
using Parameters
using Zygote
using Zygote: @adjoint

import Base: ==, +, -, *, /, ≈, size, reshape, permutedims, transpose, conj, conj!, show, similar, adjoint, copy, sqrt, getindex, setindex!, Array, broadcasted, vec, map, ndims, indexin, sum, zero
import CUDA: CuArray
import LinearAlgebra: tr, norm, normalize!, dot, rmul!, axpy!, mul!, diag, Diagonal, lmul!, axpby!, svd, svd!
import OMEinsum: _compactify!, subindex, einsum, Tr, Repeat, tensorpermute
import VectorInterface: inner, zerovector, add!!, scale!!, scale
import Zygote: accum

export AbstractSiteType, indextoqn
export randU1, zerosU1, asU1Array, asArray, getqrange, getblockdims, qndims, randU1DiagMatrix, invDiagU1Matrix, IU1
export SymmetricType
export randinitial, zerosinitial, Iinitial
export asArray, asSymmetryArray, symmetryreshape, getsymmetry, getdir
export dtr, qrpos, lqpos, qrpos!, lqpos!
export _mattype, _arraytype
export U1Array, U1reshape, U1reshapeinfo
export DoubleArray, convert_bilayer_Z2, asComplexArray, randU1double, IU1double

include("sitetype.jl")
include("base.jl")
include("utils.jl")
include("initial.jl")
include("convert.jl")
include("einsum.jl")
include("vectorinterface.jl")
include("decomposition.jl")
include("u1reshape.jl")
include("symmetrictype.jl")
include("autodiff.jl")
include("doublepeps/base.jl")
include("doublepeps/convert.jl")
include("doublepeps/initial.jl")
include("doublepeps/decompsition.jl")
include("doublepeps/utils.jl")
include("doublepeps/vectorinterface.jl")

end
