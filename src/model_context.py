from dataclasses import dataclass

from parameters import HeatTransferParameters, KineticParameters


@dataclass(frozen=True)
class PropertyModel:
    """Interface for the Le et al. physical-property library."""

    def rho_liquid(self, T):
        raise NotImplementedError

    def mu_liquid(self, T):
        raise NotImplementedError

    def sigma_liquid(self, T):
        raise NotImplementedError

    def k_liquid(self, T):
        raise NotImplementedError

    def cp_liquid(self, T):
        raise NotImplementedError

    def mu_gas(self, T, y):
        raise NotImplementedError

    def k_gas(self, T, y):
        raise NotImplementedError

    def cp_gas(self, T, y):
        raise NotImplementedError


@dataclass(frozen=True)
class ThermoModel:
    """Interface for thermodynamic relations used by the reference model."""

    def dimensionless_K(self, T):
        raise NotImplementedError

    def reaction_enthalpy(self, T):
        raise NotImplementedError


@dataclass(frozen=True)
class ModelContext:
    properties: PropertyModel
    thermo: ThermoModel
    kinetics: KineticParameters = KineticParameters()
    heat_transfer: HeatTransferParameters = HeatTransferParameters()
