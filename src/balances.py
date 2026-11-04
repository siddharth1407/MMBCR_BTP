
import numpy as np

from fluxes import calculate_fluxes
from parameters import G, M_AR, M_CH4, M_H2
from state import IDX, N_STATE


def source_terms(h, y, ctx, params):
    f = calculate_fluxes(h, y, ctx, params)

    nu = np.array([-1.0, 2.0, 0.0])
    M = np.array([M_CH4, M_H2, M_AR])

    # Eq. (2) as printed in the paper uses a minus sign with a signed
    # stoichiometric factor. With the paper's stated nu_CH4=-1 and the
    # positive forward reaction rate, that printed combination would make
    # methane increase along the reactor. For the physically consistent
    # implementation used here (and the decreasing CH4 conversion shown by
    # the authors' outputs), the reaction source is nu_i*a_G*r_A*M_i.
    source_species = nu * f.a_g * f.r_A * M

    delta_hr = ctx.thermo.reaction_enthalpy(y[IDX.T_G])

    # Sign convention follows the rendered/web copy of Eq. (11):
    # d(u_G*rho_G*Cp_G*T_G)/dh
    #   = -U_h*a_s*(T_G-T_L) - a_G*DeltaH_R*r_A.
    gas_energy_source = (
        -f.U_h * f.a_s * (y[IDX.T_G] - y[IDX.T_L])
        -f.a_g * delta_hr * f.r_A
    )

    # Eq. (13), u_L = 0.
    rho_l = ctx.properties.rho_liquid(y[IDX.T_L])
    k_g = ctx.properties.k_gas(
        y[IDX.T_G],
        {"CH4": f.x_CH4, "H2": f.x_H2, "Ar": f.x_Ar},
    )

    liquid_source = (
        f.U_h * f.a_s * (y[IDX.T_G] - y[IDX.T_L])
        + 4.0 * ctx.heat_transfer.Uw / params.D
        * (params.T_W - y[IDX.T_L])
        + 4.0 * ctx.heat_transfer.Nu_orf * k_g
        * (y[IDX.T_G] - y[IDX.T_L]) / params.D**2
    )

    pressure_gradient = -(
        rho_l * (1.0 - f.a_g) + f.rho_g * f.a_g
    ) * G

    return source_species, gas_energy_source, liquid_source, pressure_gradient


def derivative_matrix_rhs(h, y, ctx, params, eps_rel=1e-6):
    """First-order representation of the coupled BVP.

    State:
      [J_CH4, dx_CH4/dh, J_H2, dx_H2/dh,
       J_Ar,  dx_Ar/dh, P, T_G, T_L, dT_L/dh]

    Primary state variables J_i are the species mass fluxes rho*u*x_i.
    This is important because it makes u_G recoverable as
    u_G = (sum_i J_i)/rho_G, exactly matching the paper's formulation
    in terms of rho_G*u_G*x_i.

    Three species equations are second-order in composition, so the
    z_i = dx_i/dh variables are retained. The gas composition obeys
    x_CH4 + x_H2 + x_Ar = 1, so only two independent composition
    derivative constraints are required.
    """
    y = np.asarray(y, dtype=float)

    f = calculate_fluxes(h, y, ctx, params)
    species_src, gas_src, liquid_src, p_src = source_terms(
        h, y, ctx, params
    )

    # K = rho_G * alpha_G * D_ax in Eq. (2).
    K_disp = f.rho_g * f.a_g * f.D_ax

    z = np.array([
        y[IDX.ZX_CH4],
        y[IDX.ZX_H2],
        y[IDX.ZX_Ar],
    ])

    # Flux vector for numerical differentiation:
    # [J_CH4, J_H2, J_Ar, K*z_CH4, K*z_H2, K*z_Ar,
    #  gas_energy, liquid_conduction]
    def aux_vector(h_local, y_local):
        fl = calculate_fluxes(h_local, y_local, ctx, params)
        Kloc = fl.rho_g * fl.a_g * fl.D_ax
        zloc = np.array([
            y_local[IDX.ZX_CH4],
            y_local[IDX.ZX_H2],
            y_local[IDX.ZX_Ar],
        ])
        return np.concatenate([
            fl.species_mass_flux,
            Kloc * zloc,
            np.array([
                fl.gas_energy_flux,
                fl.liquid_conduction_flux,
            ]),
        ])

    base_flux = aux_vector(h, y)
    n_aux = base_flux.size
    J = np.zeros((n_aux, N_STATE))

    nonnegative_indices = {IDX.J_CH4, IDX.J_H2, IDX.J_Ar, IDX.P, IDX.T_G, IDX.T_L}
    for j in range(N_STATE):
        scale = max(abs(y[j]), 1.0)
        step = eps_rel * scale
        # Keep physically constrained states nonnegative during numerical
        # differentiation. At zero, use a one-sided derivative.
        if j in nonnegative_indices:
            if y[j] <= step:
                yp = y.copy()
                yp[j] += step
                fp = aux_vector(h, yp)
                J[:, j] = (fp - base_flux) / step
                continue
            step = min(step, 0.25 * y[j])

        yp = y.copy()
        ym = y.copy()
        yp[j] += step
        ym[j] -= step
        J[:, j] = (
            aux_vector(h, yp) - aux_vector(h, ym)
        ) / (2.0 * step)

    # Explicit h derivative, important because D_ax includes h/H smoothing.
    step_h = eps_rel * max(abs(h), 1.0)
    h_plus = min(params.H, h + step_h)
    h_minus = max(0.0, h - step_h)
    if h_plus == h_minus:
        h_plus = min(params.H, h + max(step_h, 1.0e-8))
        h_minus = max(0.0, h - max(step_h, 1.0e-8))

    Fh = (
        aux_vector(h_plus, y) - aux_vector(h_minus, y)
    ) / (h_plus - h_minus)

    # Unknown derivatives are y'[0:10].
    # Equations 0-2: J_i' - d(K*z_i)/dh = source_i.
    # These occupy rows 0-3 of [J, K*z] derivative system.
    A = np.zeros((N_STATE, N_STATE))
    b = np.zeros(N_STATE)

    # First three species balances:
    # derivative(J_i) - derivative(K*z_i) = source_i
    A[0:3, :] = J[0:3, :] - J[3:6, :]
    b[0:3] = species_src - (Fh[0:3] - Fh[3:6])

    row = 3

    # Composition derivative identities:
    # z_i = (J_i' - x_i * J_total') / J_total.
    J_total = float(f.species_mass_flux.sum())
    x = np.array([f.x_CH4, f.x_H2, f.x_Ar])

    # CH4
    A[row, IDX.J_CH4] = 1.0
    A[row, IDX.J_H2] = -x[0]
    A[row, IDX.J_Ar] = -x[0]
    b[row] = J_total * y[IDX.ZX_CH4]
    row += 1

    # H2
    A[row, IDX.J_H2] = 1.0
    A[row, IDX.J_CH4] = -x[1]
    A[row, IDX.J_Ar] = -x[1]
    b[row] = J_total * y[IDX.ZX_H2]
    row += 1

    # Ar
    A[row, IDX.J_Ar] = 1.0
    A[row, IDX.J_CH4] = -x[2]
    A[row, IDX.J_H2] = -x[2]
    b[row] = J_total * y[IDX.ZX_Ar]
    row += 1

    # Pressure equation.
    A[row, IDX.P] = 1.0
    b[row] = p_src
    row += 1

    # Gas energy equation: d(E_G)/dh = gas_src.
    A[row, :] = J[6, :]
    b[row] = gas_src - Fh[6]
    row += 1

    # Liquid conductive flux equation: d(Q_L)/dh = liquid_src.
    A[row, :] = J[7, :]
    b[row] = liquid_src - Fh[7]
    row += 1

    # Liquid-temperature definition: T_L' = zT_L.
    A[row, IDX.T_L] = 1.0
    b[row] = y[IDX.ZT_L]
    row += 1

    assert row == N_STATE

    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError(
            "Coupled MMBCR derivative system is singular. "
            "Check the property library, initial state, or constitutive "
            "relations."
        ) from exc


def model_rhs(h, y, ctx, params):
    y = np.asarray(y)

    if y.ndim == 1:
        return derivative_matrix_rhs(h, y, ctx, params)

    out = np.zeros_like(y, dtype=float)
    for k, hk in enumerate(np.atleast_1d(h)):
        out[:, k] = derivative_matrix_rhs(
            float(hk), y[:, k], ctx, params
        )
    return out
