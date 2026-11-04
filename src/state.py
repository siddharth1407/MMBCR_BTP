
from dataclasses import dataclass


@dataclass(frozen=True)
class StateIndex:
    # J_i = rho_G * u_G * x_i
    J_CH4: int = 0
    ZX_CH4: int = 1       # d x_CH4 / dh
    J_H2: int = 2
    ZX_H2: int = 3        # d x_H2 / dh
    J_Ar: int = 4
    ZX_Ar: int = 5        # d x_Ar / dh
    P: int = 6
    T_G: int = 7
    T_L: int = 8
    ZT_L: int = 9         # d T_L / dh


IDX = StateIndex()
N_STATE = 10
