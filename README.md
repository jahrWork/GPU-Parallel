# GPU-Parallel

To use the package, first you need to install julia using juliaup. For Windows users, this can be done from the Microsoft Store.
After the instalation of Julia, it is highly advisable to install the following packages in the global environment:

```julia
using Pkg
Pkg.add("Revise")
Pkg.add("BenchmarkTools")
Pkg.add("ProfileView")
```

Then, to activate the local environment and install the necessary packages, in a Julia REPL

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```
