
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_bvp

from balances import model_rhs
from boundary_conditions import bc
from state import N_STATE


@dataclass
class BVPSolution:
    result: object

    @property
    def converged(self):
        return bool(self.result.success)

    @property
    def message(self):
        return str(self.result.message)


def default_initial_guess(params, n_points=120):
    h = np.linspace(0.0, params.H, n_points)
    y = np.zeros((N_STATE, n_points))

    # Use the exact inlet mass-flux values as an initial profile.
    y[0, :] = params.j_CH4_in
    y[2, :] = params.j_H2_in
    y[4, :] = params.j_Ar_in

    # Approximate initial composition gradients as zero.
    y[1, :] = 0.0
    y[3, :] = 0.0
    y[5, :] = 0.0

    # Initial pressure decreases slightly from inlet to outlet as a numerical
    # starting guess only; it is not a published result.
    y[6, :] = np.linspace(params.P_in, 0.95 * params.P_in, n_points)

    # Start gas at inlet temperature and let the BVP determine the profile.
    y[7, :] = np.linspace(
        params.T_G_in, params.T_L_in, n_points
    )

    y[8, :] = params.T_L_in
    y[9, :] = 0.0

    return h, y


def solve_reference_model(
    ctx,
    params,
    n_points=120,
    max_nodes=4000,
    tol=1e-4,
):
    h, y0 = default_initial_guess(params, n_points=n_points)

    result = solve_bvp(
        fun=lambda h, y: model_rhs(h, y, ctx, params),
        bc=lambda ya, yb: bc(ya, yb, ctx, params),
        x=h,
        y=y0,
        tol=tol,
        max_nodes=max_nodes,
        verbose=1,
    )

    return BVPSolution(result=result)
