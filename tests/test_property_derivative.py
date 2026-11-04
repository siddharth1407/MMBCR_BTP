from properties import species_heat_capacity_molar, species_heat_capacity_molar_derivative


def test_species_cp_derivative_matches_finite_difference():
    for s, T in (("CH4", 1100.0), ("H2", 1200.0), ("Ar", 1300.0)):
        eps = 1e-2
        fd = (species_heat_capacity_molar(s, T + eps) - species_heat_capacity_molar(s, T - eps)) / (2 * eps)
        analytic = species_heat_capacity_molar_derivative(s, T)
        assert abs(fd - analytic) < 1e-7
