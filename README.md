# U1ArrayKit.jl

[![Build Status](https://github.com/XingyuZhang2018/U1ArrayKit.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/XingyuZhang2018/U1Array.jl/actions/workflows/CI.yml?query=branch%3Amaster)

This package provides a set of tools for manipulating U(1) arrays in Julia. The main type is `U1Array`, which is a subtype of `AbstractArray{ComplexF64, N}`. 

Also include Z2 symmetry by setting `ifZ2 = true` in the constructor.