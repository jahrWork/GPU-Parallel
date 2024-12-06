using Pkg

# Project.toml path, where the packages are listed
project_path = pwd()

println("Activating the project environment...")
println(" ")
Pkg.activate(project_path)

println(" ")

println("Installing the packages according to [deps]...")
println(" ")
Pkg.instantiate()

println(" ")

println("Updating the packages according to [compat]...")
println(" ")
Pkg.resolve()

println(" ")

println("Versions of the packages installed:")
println(" ")
Pkg.status()

println(" ")

println("Project environment ready.")
