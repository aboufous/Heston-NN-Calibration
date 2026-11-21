# Installe les paquets Julia listés dans julia/Project.toml (une seule fois).  Usage : julia julia/setup.jl
using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
