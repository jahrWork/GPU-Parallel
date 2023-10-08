using LinearAlgebra, MKL, DelimitedFiles
# Change between BLISBLAS and MKL

# To use parallel loops with all threads:
# · Open VS Code Settings
# · Look for Julia Threads and change the settings.json

BLAS.get_config()
BLAS.set_num_threads(1)

function ab_multiply!(C, A, B, TIMES)
    # Use parallel loop by switching the next two lines
    # Threads.@threads for i ∈ 1:TIMES
    for i ∈ 1:TIMES
        mul!(C, A, B) # gemm
    end
end

N_ops = 2 * 10000^3

N = 5000
TIMES = 2

A = rand(Float32, N, N)
B = rand(Float32, N, N)
C = similar(B)

ab_multiply!(C, A, B, TIMES)

computation_times = Float64[]

N_list_1 = 50:25:2500
TIMES_list = 63:-1:1

# Second list stores N that is calculated in TIMES loop
N_list_2 = Int32[]



for N in N_list_1

    TIMES = ceil(Int, N_ops / (2 * N^3))

    A = rand(Float32, N, N)
    B = rand(Float32, N, N)
    C = similar(B)

    push!(computation_times, @elapsed ab_multiply!(C, A, B, TIMES))
    @show N
    sleep(5)
end

for TIMES in TIMES_list

    N = ceil(Int, (N_ops / (2 * TIMES))^(1/3))
    push!(N_list_2, N)
    A = rand(Float32, N, N)
    B = rand(Float32, N, N)
    C = similar(B)

    push!(computation_times, @elapsed ab_multiply!(C, A, B, TIMES))
    @show N
    sleep(5)
end

N_list = vcat(N_list_1, N_list_2)

# writedlm("data/julia_mkl_sc.csv", [N_list computation_times], ',')
writedlm("data/julia_mkl_mc.csv", [N_list computation_times], ',')
# writedlm("data/julia_blis_mc.csv", [N_list computation_times], ',')

# Make sure you have a "data" folder inside the working folder
