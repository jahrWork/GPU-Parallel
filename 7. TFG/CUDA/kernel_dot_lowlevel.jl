using CUDA
CUDA.allowscalar(false)
using BenchmarkTools
using LinearAlgebra

N = 2050
a = CUDA.rand(Float32, N)
b = CUDA.rand(Float32, N)
c = CUDA.zeros(Float32, 1)

typeof(a) <: AbstractArray

function my_dot_kernel_lowlevel!(dot_pointer, a, b)
    i = CUDA.threadIdx().x + (CUDA.blockIdx().x - 1) * CUDA.blockDim().x
    if i <= length(a)
       # https://cuda.juliagpu.org/stable/development/kernel/#Atomics

       #  Access the element at 'i' and get its raw memory address.
        a_val_pointer = CUDA.pointer(a, i) # Get the pointer to the i-th element of 'a'.
        b_val_pointer = CUDA.pointer(b, i) # Get the pointer to the i-th element of 'b'.

        # Get the value at the memory location.
        a_val = unsafe_load(a_val_pointer) # Load the value of the i-th element of 'a'
        b_val = unsafe_load(b_val_pointer) # Load the value of the i-th element of 'b'

        # Perform the low-level atomic operation, using matching types
        CUDA.atomic_add!(dot_pointer, a_val * b_val) # Atomic addition using a memory pointer
    end
    return nothing
end

function my_dot_custom_lowlevel!(c::CuArray, a::CuArray, b::CuArray)
    dot_pointer = CUDA.pointer(c) # Get the pointer to the first (and only) element in 'c'
    @cuda blocks=cld(length(a), 1024) threads=1024 my_dot_kernel_lowlevel!(dot_pointer, a, b)
    return nothing
end

my_dot_custom_lowlevel!(c, a, b)

# Check if the addition is correct
isapprox(Array(c)[1], dot(Array(a), Array(b)))