import numpy as np
from parameters import M_CH4, M_H2, M_AR


def test_physical_reaction_mass_source_is_mass_decreasing():
    a = 0.003
    r = 5.0
    src = np.array([-1.0, 2.0, 0.0]) * a * r * np.array([M_CH4, M_H2, M_AR])
    assert src[0] < 0
    assert src[1] > 0
    assert np.isclose(src.sum(), a*r*(-M_CH4 + 2*M_H2))
