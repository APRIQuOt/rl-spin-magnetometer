"""
++ Hamiltonians ++

System Hamiltonians and utilities
"""

export H_pulsed

# ----- Utilities -----
"""
    heaviside(t)

Heaveyside function, taking on value of 1 for t>0, 0 for t<0.
"""
function heaviside(t)
    (sign(t) + 1) / 2
end
 

"""
    square(t, a, b)

Square pulse; function returns 1 for a<t<b, and 0 elsewhere.

# Arguments
- `t`: Point at which to evaluate pulse.
- `a`: Start of square pulse.
- `b`: End of square pulse.
"""
function square(t, a, b)
    heaviside(t-a) - heaviside(t-b)
end


"""
    H_pulsed(t; F, ωz, Ωs, ϕs, ωs, ts, TΩ)

Hamiltonian for N pulses with variable amplitude, phase, and carrier freq.

# Arguments
- `t`: time argument, needed for solvers.
- `F`: Vector of spin matrices, (Fx, Fy, Fz), as from JuliaUtils.spin_matrices
- `ωz`: Spin precession freq.
- `Ωs`: List of kicking strengths.
- `ϕs`: List of kick phases.
- `δs`: List of kick detunings.
- `ts`: List of kicking times.
- `TΩ`: Kick duration.
"""
function H_pulsed(t; F, ωz, Ωs, ϕs, δs, ts, TΩ)
    H = ωz * F[3]
    for (Ω, ϕ, δ, tk) in zip(Ωs, ϕs, δs, ts)
        H += Ω * square(t, tk, tk + TΩ) * (sin(ϕ) * F[1] - cos(ϕ) * F[2])
        H += -1 * square(t, tk, tk + TΩ) * δ * F[3]
    end
    H
end
