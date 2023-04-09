import astropy.units as u
import natpy as nat

@u.quantity_input
class colapsar():
    def __init__(self,mass:nat.Msun,temperature:nat.MeV,
                 chemical_potential_e:nat.MeV,n_eff_N:nat.MeV**(3)):

        self._mass                  = mass
        self._temperature           = temperature
        self._chemical_potenitial_e = chemical_potential_e
        self._n_eff_N               = n_eff_N

        return
    
    @property
    def mass(self):
        return self._mass
    
    @property
    def temperature(self):
        return self._temperature
    
    @property
    def mu_e(self):
        return self._chemical_potenitial_e
    
    @property
    def n_eff_N(self):
        return self._n_eff_N
    
    @mass.setter
    @u.quantity_input
    def mass(self,thismass:nat.Msun):
        self._mass = thismass

        return
    
    @temperature.deleter
    @u.quantity_input
    def temperature(self,temp:nat.MeV):
        self._temperature = temp

        return
    
    @mu_e.setter
    @u.quantity_input
    def mu_e(self,thismu_e:nat.MeV):
        self._chemical_potenitial_e = thismu_e

        return
    
    @n_eff_N.setter
    @u.quantity_input
    def n_eff_N(self,n_eff:nat.MeV**(3)):
        self._n_eff_N = n_eff

        return
    