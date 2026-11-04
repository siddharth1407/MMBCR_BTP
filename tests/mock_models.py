
class MockProperties:
    """Numerical plumbing test only — NOT physical Ni-Bi data."""

    def rho_liquid(self, T):
        return 6500.0

    def mu_liquid(self, T):
        return 1.0e-3

    def sigma_liquid(self, T):
        return 0.5

    def k_liquid(self, T):
        return 30.0

    def cp_liquid(self, T):
        return 300.0

    def mu_gas(self, T, y):
        return 5.0e-5

    def k_gas(self, T, y):
        return 0.10

    def cp_gas(self, T, y):
        return 1800.0


class MockThermo:
    """Numerical plumbing test only — NOT paper thermodynamics."""

    def dimensionless_K(self, T):
        return 1.0e3

    def reaction_enthalpy(self, T):
        return 3.7e4
