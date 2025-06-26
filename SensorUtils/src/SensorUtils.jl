"""
++ SensorUtils ++

Combined package, containing wrappers to call simulations from Python via JuliaCall
"""

module SensorUtils

export apply_pulses, qfi_pulsed


# Include functions
include("Simulations.jl")  # TDSE & GKSL solvers, plus QFI
include("Hamiltonians.jl")  # Functions specific to spin-magnetometer


# Wrappers
"""
    apply_pulses(ρ, t_ax, spin, ωz, Ωs, ϕs, δs, ts, TΩ, Ls, γs)

Wrapper to simulate `H_pulsed` evolution for times in `t_ax`.

# Arguments
- `ρ`: Initial density matrix.
- `t_ax`: Times at which to evaluate density matrix.
- `spin`: spin quantum number.
- `ωz`: Spin precession freq.
- `Ωs`: List of kicking strengths.
- `ϕs`: List of kick phases.
- `δs`: List of kick detunings.
- `ts`: List of kicking times.
- `TΩ`: Kick duration.
- `Ls`: List of jump operators, defaults to list with Identity.
- `γs`: List of decay rates corresponding to jump operators `Ls`.
"""
function apply_pulses(ρ, t_ax, spin, ωz, Ωs, ϕs, δs, ts, TΩ, Ls, γs)
    F = spin_matrices(rationalize(spin))
    ρ_t = solve_for_rho(H_pulsed, ρ, (0, t_ax[end]); F, ωz, Ωs, ϕs, δs, ts, TΩ, Ls, γs)
    [ρ_t(t) for t in t_ax]
end


"""
    qfi_pulsed(ρ0, t_ax, spin, ωz, Ωs, ϕs, δs, ts, TΩ, Ls, γs)

Wrapper to compute QFI for `H_pulsed` evolution at times in `t_ax`.

# Arguments
- `ρ0`: Initial density matrix.
- `t_ax`: Times at which to evaluate density matrix.
- `spin`: spin quantum number.
- `ωz`: Spin precession freq.
- `Ωs`: List of kicking strengths.
- `ϕs`: List of kick phases.
- `δs`: List of kick detunings.
- `ts`: List of kicking times.
- `TΩ`: Kick duration.
- `Ls`: List of jump operators, defaults to list with Identity.
- `γs`: List of decay rates corresponding to jump operators `Ls`.
"""
function qfi_pulsed(ρ0, t_ax, spin, ωz, Ωs, ϕs, δs, ts, TΩ, Ls, γs)
    F = spin_matrices(rationalize(spin))
    [qfi(ρ0, (:ωz, ωz), t; H=H_pulsed, F, Ωs, ϕs, δs, ts, TΩ, Ls, γs) for t in t_ax]
end

end # module SensorUtils
