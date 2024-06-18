using Pkg
Pkg.add("MKL")
using MKL
using LinearAlgebra

# Verificar si MKL está habilitado
println("BLAS vendor: ", LinearAlgebra.BLAS.vendor())

# Generar el ensamblador para la función `dot`
A = rand(Float32, 100)
B = rand(Float32, 100)
@code_native dot(A, B)

# Listar las bibliotecas dinámicas cargadas
using Libdl
for (i, lib) in enumerate(Libdl.dllist())
    println("Library $i: ", lib)
end

