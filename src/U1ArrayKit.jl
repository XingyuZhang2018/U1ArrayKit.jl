module U1ArrayKit

using CUDA
using LinearAlgebra
using OMEinsum
using Parameters
using Zygote

import Base: ==, +, -, *, /, ≈, size, reshape, permutedims, transpose, conj, conj!, show, similar, adjoint, copy, sqrt, getindex, setindex!, Array, broadcasted, vec, map, ndims, indexin, sum, zero
import CUDA: CuArray
import LinearAlgebra: tr, norm, dot, rmul!, axpy!, mul!, diag, Diagonal, lmul!, axpby!, svd, svd!
import OMEinsum: _compactify!, subindex, einsum, Tr, Repeat, tensorpermute
import VectorInterface: inner, zerovector, add!!, scale!!, scale
import Zygote: accum

export AbstractSiteType, indextoqn
export randU1, zerosU1, asU1Array, asArray, getqrange, getblockdims, qndims, randU1DiagMatrix, invDiagU1Matrix, IU1
export SymmetricType
export asArray, asSymmetryArray, symmetryreshape, getsymmetry, getdir
export qrpos, lqpos
export _mattype, _arraytype
export randinitial, zerosinitial, Iinitial
export U1Array, U1reshape, U1reshapeinfo

include("sitetype.jl")
include("base.jl")
include("initial.jl")
include("convert.jl")
include("symmetrictype.jl")
include("einsum.jl")
include("vectorinterface.jl")
include("decomposition.jl")
include("u1reshape.jl")

end
