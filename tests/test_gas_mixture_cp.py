import math
from properties import gas_mixture_heat_capacity, species_heat_capacity_molar
from parameters import M_CH4, M_AR

def test_gas_mixture_cp_uses_molar_mixing_mass_conversion():
    T=1073.15
    y={"CH4":0.8*M_CH4/(0.8*M_CH4+0.2*M_AR),"H2":0.0,"Ar":0.2*M_AR/(0.8*M_CH4+0.2*M_AR)}
    cp=gas_mixture_heat_capacity(T,y)
    xch4=0.8; xar=0.2
    cp_molar=xch4*species_heat_capacity_molar('CH4',T)+xar*species_heat_capacity_molar('Ar',T)
    Mmix=xch4*M_CH4+xar*M_AR
    expected=cp_molar/Mmix
    assert math.isclose(cp, expected, rel_tol=1e-12)
