"""
++ Simulations ++

Functions for integrating Hamiltonian, computing the QFI, and related utilities.
"""

using OrdinaryDiffEq, StaticArrays, ForwardDiff, LinearAlgebra, Integrals
    

export spin_matrices, spin_matrices_pm, solve_for_rho, qfi_lyapunov, qfi_safranek, qfi,
    qfi_max


# ----- Helper Functions -----
δ(x, y) = x == y  # Kronecker delta
comm = (A, B) -> A*B - B*A  # Commutator
anticomm = (A, B) -> A*B + B*A  # Anti-commutator


"""
    ∂ₜ(f)

Just a little helper function that turns `ForwardDiff.derivative` into a higher order
function. This uses 'automatic differentiation' rather than finite differences.

# Example
```julia-repl
    julia> ∂ₜ(cos)(1.0) == -sin(1.0)
    true
```
"""
∂ₜ(f) = t -> ForwardDiff.derivative(f, t)


"""
    spin_matrices(s::Rational)

Returns a (static) vector of spin-s (static) matrices `[Sx, Sy, Sz]` which are `2s+1`
dimensional, and obey the spin algebra

    S[i]*S[j] - S[j]*S[i] ≈ sum(k -> 𝒾*ϵ[i,j,k]*S[k], 1:3)

where `ϵ[i,j,k]` is the Levi-Civita symbol. Note that `s` must be positive half-integer
valued.
"""
spin_matrices(s::Union{Int,Rational}) = spin_matrices(Val{s}) #lift to type domain

@generated function spin_matrices(::Type{Val{s}}) where {s}
    denominator(2s + 1) == 1 && s >= 0 || throw("s must be positive half-integer valued.
        Got $s")
    d = Int(2s + 1)
    Sx = (1 / 2 * (δ(a, b + 1) + δ(a + 1, b)) * √((s + 1) * (a + b - 1) - a * b)
        for a ∈ 1:d, b ∈ 1:d)
    Sy = (im / 2 * (δ(a, b + 1) - δ(a + 1, b)) * √((s + 1) * (a + b - 1) - a * b)
        for a ∈ 1:d, b ∈ 1:d)
    Sz = (δ(a, b) * (s + 1 - a) for a ∈ 1:d, b ∈ 1:d)

    :(SA[SMatrix{$d,$d}($(Sx...)),
        SMatrix{$d,$d}($(Sy...)),
        SMatrix{$d,$d}($(Sz...))])
end


"""
    spin_matrices_pm(s)

Helper function to return F+ and F- matrices, for spin manifold `s`.

# Arguments
- `s`: Spin manifold, must be a positive half-integer.
"""
function spin_matrices_pm(s)
    F = spin_matrices(s)
    Fp = F[1] + im * F[2]
    Fm = F[1] - im * F[2]
    return Fp, Fm
end


# ----- Solver Functions -----
"""
    ∂ₜρ(H, Ls, γs)

Define an ODE specifying the GKSL master equation, from Hamiltonian and jump operators.

This is a higher order function, designed to be passed to an ODE solver from
OrdinaryDiffEq.jl. Hence, for Hamiltonian function `H`, and lists of the jump operators
`Ls` with corresponding decay rates `γs`, this returns the function

    (ρ, p, t) -> -im * comm(H(t; p...), ρ) + jump_ops(Ls, γs, ρ)

with `ρ` the density matrix, `p` the Hamiltonian parameters, and `t` time. The function
`jump_ops` computes the decay terms for the GKSL equation.

# Arguments
- `H`: Hamiltonian function.
- `Ls`: List of Lindbladian 'jump operators'.
- `γs`: List of decay rates corresponding to jump operators `Ls`.
"""
∂ₜρ(H, Ls, γs) = (ρ, p, t) -> -im * comm(H(t; p...), ρ) + jump_ops(Ls, γs, ρ)


"""
    jump_ops(Ls, γs, ρ)

Unpacks list of jump operators `Ls` and corresponding decay rates `γs` for use in
GKSL solver.

# Arguments
- `Ls`: Lindbladian jump operators.
- `γs`: Decay rates corresponding to jump operators.
- `ρ`: Density matrix.
"""
function jump_ops(Ls, γs, ρ)
    Lop = zeros(ComplexF64, size(ρ))
    for (γ, L) in zip(γs, Ls)
        Lop += γ * (L * ρ * adjoint(L) - (1/2) * anticomm(adjoint(L) * L, ρ))
    end
    Lop
end


"""
    solve_for_rho(H, ρ0, tspan, alg=Tsit5(); Ls=[I], γs=[0.], reltol=1e-8,
        abstol=1e-8, maxiters=10_000_000, kwargs...)

Approximately solve the GKSL equation.

This returns a callable object `ρ` which satisfies the GKSL equation for the given
Hamiltonian between the lower and upper bounds on `tspan`. If you call `ρ(t)` where
`t` is *outside* of the bounds in `tspan`, you will get a nonsense result.

# Arguments:
- `H`: Function specifying the Hamiltonian.
- `ρ0`: Initial density matrix.
- `tspan`: a container whose first element should be the starting time and second element
    should be the ending time for the solver.
- `alg` (default `Tsit5()`): Determines the ODE solver algorithm with which to solve the
    GKSL equation.
- `Ls` (default `[I]`): List of jump operators, defaults to list with Identity.
- `γs` (default `[0.]`): List of decay rates corresponding to jump operators `Ls`.
- `reltol` (default `1e-8`): Desired relative numerical accuracy
- `abstol` (default `1e-8`): Desired absolute numerical accuracy
- `maxiters` (default `10_000_000`): Maxumum number of iterations in the differential
    equation solving algorithm before it gives up.
- `kwargs...` All remaining keyword arguments will be passed to the Hamiltonian
    function `H`.
"""
function solve_for_rho(H, ρ0, tspan, alg=Tsit5(); Ls=[I], γs=[0.],
    reltol=1e-8, abstol=1e-8, maxiters=10_000_000, kwargs...)
    prob = ODEProblem(∂ₜρ(H, Ls, γs), ρ0, tspan, (;kwargs...))
    solve(prob, alg; reltol, abstol, maxiters)
end


# ----- QFI -----
"""
    qfi_lyapunov(ρ0, θ::Tuple{Symbol, Number}, t; H, kwargs...)

Compute the quantum Fisher information through the Lyapunov integral representation.

# Arguments
- `ρ0`: Initial density matrix.
- `θ`: Parameter to estimate QFI from, as a `NamedTuple`. Symbol must match a parameter
    of `H`.
- `t`: Time at which to evaluate the QFI.
- `H`: Hamiltonian function describing parameterized evolution under the GKSL equation.
- `abstol` (default `1e-4`): Absolute tolerance for SLD integral.
- `reltol` (default `1e-4`): Relative tolerance for SLD integral.
- `kwargs`: *Other* kwargs passed to `H`, should not contain `θ`.
"""
function qfi_lyapunov(ρ0, θ::Tuple{Symbol, Number}, t; H, abstol=1e-4, reltol=1e-4,
    kwargs...)
    # Differentiable Density Matrix
    key, val = θ
    ρ = θ -> solve_for_rho(H, ρ0, (0, t); key=>θ, kwargs...)(t)  # DM
    dρ = ForwardDiff.derivative(ρ, val)  # dρ/dθ @ (θ = θ)

    # SLD
    dL = (s::Number, θ::Number) -> 2 * exp(-1 * s * ρ(θ)) * dρ * exp(-1 * s * ρ(θ))
    prob = IntegralProblem(dL, (0., Inf), val)
    L = solve(prob, QuadGKJL(); reltol, abstol)
    real(tr(ρ(val) * L.u^2))  # QFI
end


"""
    qfi_safranek(ρ0, θ::Tuple{Symbol, Number}, t; H, kwargs...)

Compute the quantum Fisher information from the Šafránek representation.

# Arguments
- `ρ0`: Initial density matrix.
- `θ`: Parameter to estimate QFI from, as a `NamedTuple`. Symbol must match a parameter
    of `H`.
- `t`: Time at which to evaluate the QFI
- `H`: Hamiltonian function describing parameterized evolution under the GKSL equation.
- `kwargs`: *Other* kwargs passed to `H`, should not contain `θ`.
"""
function qfi_safranek(ρ0, θ::Tuple{Symbol, Number}, t; H, kwargs...)
    key, val = θ
    ρ = θ -> solve_for_rho(H, ρ0, (0, t); key=>θ, kwargs...)(t)  # DM
    dρ = ForwardDiff.derivative(ρ, val)  # dρ/dθ @ (θ = θ)

    Id = Matrix{Float64}(I, size(ρ0)...)
    real(2 * adjoint(vec(dρ)) * inv(kron(conj(ρ(val)), Id) + kron(Id, ρ(val))) * vec(dρ))
end


"""
    qfi(ρ0, θ, t; H, kwargs...)

Compute QFI using Šafránek method if possible, otherwise Lyapunov.

# Arguments
- `ρ0`: Initial density matrix.
- `θ`: Parameter to estimate QFI from, as a `NamedTuple`. Symbol must match a parameter
    of `H`.
- `t`: Time at which to evaluate the QFI
- `H`: Hamiltonian function describing parameterized evolution under the GKSL equation.
- `kwargs`: *Other* kwargs passed to `H`, should not contain `θ`.
"""
function qfi(ρ0, θ::Tuple{Symbol, Number}, t; H, abstol=1e-1, reltol=1e-1, kwargs...)
    QFI = try
        qfi_safranek(ρ0, θ::Tuple{Symbol, Number}, t; H, kwargs...)
    catch
        qfi_lyapunov(ρ0, θ::Tuple{Symbol, Number}, t; H, abstol, reltol, kwargs...)
    end
    QFI
end
