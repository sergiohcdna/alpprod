import astropy.units as u
import natpy as nat
import numpy as np

@u.quantity_input
class alp():
    def __init__(self,mass:nat.MeV,energy:nat.MeV,
                 g_a_e,g_a_gamma):
        self._mass       = mass
        self._energy     = energy
        self._g_ae       = g_a_e
        self._g_agamma   = g_a_gamma
        self._four_p     = self._get_4_momentum(self._energy,self._mass)
        self._norm_vec_p = self._get_norm_3_momentum(self._energy,self._mass)

        return
    
    @property
    def mass(self):
        return self._mass
    
    @property
    def energy(self):
        return self._energy
    
    @property
    def e_coupling(self):
        return self._g_ae
    
    @property
    def gamma_coupling(self):
        return self._g_agamma
    
    @property
    def four_p(self):
        return self._four_p
    
    @property
    def norm_spat_p(self):
        return self._norm_vec_p
    
    @mass.setter
    @u.quantity_input
    def mass(self,alp_mass:nat.MeV):
        self._mass = alp_mass

        # and then, I need to change the values of the momenta
        four_p = self._get_4_momentum(self._energy,alp_mass)
        spat_p = self._get_norm_3_momentum(self._energy,alp_mass)

        self._four_p      = four_p
        self._norm_spat_p = spat_p

        return
    
    @energy.setter
    @u.quantity_input
    def energy(self,alp_eng:nat.MeV):
        self._energy = alp_eng

        # and then, I need to change the values of the momenta
        four_p = self._get_4_momentum(self._energy,self._mass)
        spat_p = self._get_norm_3_momentum(self._energy,self._mass)

        self._four_p     = four_p
        self._norm_vec_p = spat_p

        return
    
    @e_coupling.setter
    def e_coupling(self,alp_gae):
        self._g_ae = alp_gae

        return
    
    @gamma_coupling.setter
    def gamma_coupling(self,alp_gagamma):
        self._g_agamma = alp_gagamma

        return
    
    @staticmethod
    @u.quantity_input
    def _get_4_momentum(energy:nat.MeV,mass:nat.MeV):
        p = [energy,mass]

        return p
    
    @staticmethod
    @u.quantity_input
    def _get_norm_3_momentum(energy:nat.MeV,mass:nat.MeV):
        norm = np.sqrt(energy.value**2-mass.value**2)

        return norm*nat.MeV
    
@u.quantity_input
class sm_fermion():
    def __init__(self,mass:nat.MeV,energy:nat.MeV,charge):
        self._mass       = mass
        self._energy     = energy
        self._charge     = charge
        self._four_p     = self._get_4_momentum(self._energy,self._mass)
        self._norm_vec_p = self._get_norm_3_momentum(self._energy,self._mass)

        return
    
    @property
    def mass(self):
        return self._mass
    
    @property
    def energy(self):
        return self._energy
        
    @property
    def four_p(self):
        return self._four_p
    
    @property
    def norm_spat_p(self):
        return self._norm_vec_p

    @property
    def charge(self):
        return self._charge

    @mass.setter
    @u.quantity_input
    def mass(self,alp_mass:nat.MeV):
        self._mass = alp_mass

        # and then, I need to change the values of the momenta
        four_p = self._get_4_momentum(self._energy,alp_mass)
        spat_p = self._get_norm_3_momentum(self._energy,alp_mass)

        self._four_p      = four_p
        self._norm_spat_p = spat_p

        return
    
    @energy.setter
    @u.quantity_input
    def energy(self,alp_eng:nat.MeV):
        self._energy = alp_eng

        # and then, I need to change the values of the momenta
        four_p = self._get_4_momentum(self._energy,self._mass)
        spat_p = self._get_norm_3_momentum(self._energy,self._mass)

        self._four_p     = four_p
        self._norm_vec_p = spat_p

        return
    
    @charge.setter
    def charge(self,q):
        self._charge = q

        return
        
    @staticmethod
    @u.quantity_input
    def _get_4_momentum(energy:nat.MeV,mass:nat.MeV):
        p = [energy,mass]

        return p
    
    @staticmethod
    @u.quantity_input
    def _get_norm_3_momentum(energy:nat.MeV,mass:nat.MeV):
        norm = np.sqrt(energy.value**2-mass.value**2)

        return norm*nat.MeV
