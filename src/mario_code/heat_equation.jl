using LinearAlgebra, DifferentialEquations, SparseArrays

#Calculates the heat equation for a given matrix using Tchebyshev's method
function heat_equation(A::Matrix{Float64}, b::Vector{Float64}, tol::Float64, max_iter::Int64)
    n = size(A, 1)
    x = zeros(n)
    x_new = zeros(n)
    r = b - A*x
    p = r
    for i in 1:max_iter
        α = dot(r, r) / dot(p, A*p)
        x_new = x + α*p
        r_new = r - α*A*p
        if norm(r_new) < tol
            return x_new
        end
        β = dot(r_new, r_new) / dot(r, r)
        p = r_new + β*p
        x = x_new
        r = r_new
    end
    return x
end

#Calculates the heat equation for a given matrix using Tchebyshev's method
const N = 32
const xyd_brusselator = range(0, stop=1, length=N)
const h = xyd_brusselator[2] - xyd_brusselator[1]
const xy_brusselator = xyd_brusselator[1:end-1]
brusselator_f = (t, u, du) -> begin
    du[1] = 0.1*(1 - u[1]) - u[1]*u[2]^2
    du[2] = 0.1*(1 - u[1]) - u[1]*u[2]^2
end

function brusselator_2d_loop(du, u, p, t) 
    A, B, alpha, dx = p
    @inbounds for i in 1:N
        for j in 1:N
            idx = (i-1)*N + j
            du[idx, 1] = 0.1*(1 - u[idx, 1]) - u[idx, 1]*u[idx, 2]^2
            du[idx, 2] = 0.1*(1 - u[idx, 1]) - u[idx, 1]*u[idx, 2]^2
        end
    end
end

#Compares the results
function test_brusselator_2d()
    A = zeros(N*N, N*N)
    B = zeros(N*N, N*N)
    alpha = 0.1
    dx = h
    for i in 1:N
        for j in 1:N
            idx = (i-1)*N + j
            A[idx, idx] = -4
            if i > 1
                A[idx, idx - N] = 1
            end
            if i < N
                A[idx, idx + N] = 1
            end
            if j > 1
                A[idx, idx - 1] = 1
            end
            if j < N
                A[idx, idx + 1] = 1
            end
            B[idx, idx] = 1
        end
    end
    u0 = ones(N*N, 2)
    p = (A, B, alpha, dx)
    prob = ODEProblem(brusselator_2d_loop, u0, (0.0, 1.0), p)
    sol = solve(prob, Tsit5(), saveat=0.1)
    @show sol
end

test_brusselator_2d()
