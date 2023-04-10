import numpy as np
import astropy.units as u
import natpy as nat

import time

import warnings

@u.quantity_input
class alp_bremsstrahlung():
    def __init__(self,alps,sm_parton_i,sm_parton_f,colapsar,
                 deltar:nat.MeV**(-1),emin:nat.MeV,emax:nat.MeV,
                 ciamin=-1,ciamax=1,cifmin=-1,cifmax=1,
                 deltamin=0,deltamax=2*np.pi,nintervals=200,
                 nsteps=500,tol=1.e-10):

        self._initial      = sm_parton_i
        self._final        = sm_parton_f
        self._alp          = alps
        self._colapsar     = colapsar
        self._prod_rate    = 0
        self._deltar       = deltar
        self._emin         = emin
        self._emax         = emax
        self._ciamin       = ciamin
        self._ciamax       = ciamax
        self._ciarange     = [ciamin,ciamax]
        self._cifmin       = cifmin
        self._cifmax       = cifmax
        self._cifrange     = [cifmin,cifmax]
        self._deltamin     = deltamin
        self._deltamax     = deltamax
        self._deltarange   = [deltamin,deltamax]
        self._nintervals   = nintervals
        engfmin            = self._emin.value
        engfmax            = self._emax.value
        self._engfmin      = engfmin
        self._engfmax      = engfmax
        self._engfstep     = (engfmax-engfmin)/self._nintervals
        self._efintervals  = [[engfmin+n*self._engfstep,engfmin+(n+1)*self._engfstep]
                              for n in range(self._nintervals)]
        self._erange       = [emin.value,emax.value]
        self._dr           = self._deltar.value
        self._ndim         = 4
        self._nsteps       = nsteps
        self._tol          = tol

        return
    
    @property
    def alp_rate(self):
        return self._prod_rate
    
    @property
    def alp_energy(self):
        return self._alp.energy
    
    @alp_energy.setter
    @u.quantity_input
    def alp_energy(self,alp_eng:nat.MeV):
        self._alp.energy = alp_eng

        return
    
    @property
    def deltar(self):
        return self._deltar
    
    @deltar.setter
    @u.quantity_input
    def deltar(self,dr:nat.MeV**(-1)):
        self._deltar = dr

        return

    def _get_fd_init(self):
        value = 0

        energy = self._initial.energy.value
        mu_pot = self._colapsar.mu_e.value
        temp   = self._colapsar.temperature.value

        if energy > 3*mu_pot:
            ratio = energy/temp
            value = np.exp(-ratio)
        else:
            if self._initial.charge<0:
                ratio = (energy-mu_pot)/temp
            else:
                ratio = (energy+mu_pot)/temp
            value = 1/(np.exp(ratio)+1)

        return value

    def _get_pauliblock(self):

        value = 0

        energy = self._final.energy.value
        mu_pot = self._colapsar.mu_e.value
        temp   = self._colapsar.temperature.value

        if energy > 3*mu_pot:
            ratio = energy/temp
            value = np.exp(-ratio)
        else:
            if self._initial.charge<0:
                ratio = (energy-mu_pot)/temp
            else:
                ratio = (energy+mu_pot)/temp
            value = 1/(np.exp(ratio)+1)

        return 1-value

    def _three_momentum_dot(self,initial,final,cif):
        pi = initial.norm_spat_p.value
        pf = final.norm_spat_p.value

        return pi*pf*cif

    def _four_momentum_dot(self,initial,final,cif):
        engi = initial.energy.value
        engf = final.energy.value
        pipf = self._three_momentum_dot(initial,final,cif)

        product = engi*engf - pipf

        return product

    def _get_cfa(self,cia,cif,delta):
        cfa = cia*cif + np.sqrt(1-cia**2)*np.sqrt(1-cif**2)*np.cos(delta)

        return cfa

    def _p_transfer(self,cif,cia,delta):
        cfa = self._get_cfa(cif,cia,delta)

        pipa = self._three_momentum_dot(self._initial,self._alp,cia)
        pipf = self._three_momentum_dot(self._initial,self._final,cif)
        pfpa = self._three_momentum_dot(self._final,self._alp,cfa)

        pinorm = self._initial.norm_spat_p.value
        panorm = self._alp.norm_spat_p.value
        pfnorm = self._final.norm_spat_p.value

        transfer = pinorm**2 +panorm**2 + pfnorm**2 + 2*pfpa -2*pipa - 2*pipf

        return np.sqrt(transfer)

    def _get_m2(self,cif,cia,delta):

        enga     = self._alp.energy.value
        engi     = self._initial.energy.value
        engf     = self._final.energy.value
        malp     = self._alp.mass.value
        charge   = self._initial.charge
        ga_sm    = self._alp.e_coupling
        n_eff    = self._colapsar.n_eff_N.value
        temp     = self._colapsar.temperature.value
        mass     = self._initial.mass.value
        deltaEaf = enga - engf
        deltaEif = engi - engf
        cfa      = self._get_cfa(cif,cia,delta)

        pipf     = self._four_momentum_dot(self._initial,self._final,cif)
        pipa     = self._four_momentum_dot(self._initial,self._alp,cia)
        pfpa     = self._four_momentum_dot(self._final,self._alp,cfa)
        transfer = self._p_transfer(cif,cia,delta)

        den1   = (malp**2+2*pfpa)**2 * (malp**2-2*pipa)**2
        val1   = ga_sm**2 * charge**4 * n_eff * temp
        val2   = transfer**2 * (temp*transfer**2 + n_eff*charge**2)
        factor = val1/val2

        term1 = pfpa**2*(4*pipa**2 + malp**2*(pipf-mass**2+2*engi*deltaEaf) 
                        - 4*pipa*(malp**2+enga*deltaEif))
        term2 = 2*pfpa*(2*pipa**2*(malp**2+enga*deltaEif)-pipa**3
                        - malp**2*(malp**2-enga**2)*(mass**2-pipf)
                        - pipa*(pipf*(malp**2-2*enga**2)+malp**4
                                - mass**2*malp**2+malp**2*(3*enga*deltaEif-2*engi*engf)
                                + 2*mass**2*enga**2)
                        + malp**4*engi*(enga-2*engf))
        term3 = malp**2*(-pipa**2*(mass**2+2*engf*(enga+engi)-pipf) 
                        - malp**2*(malp**2-enga**2)*(mass**2-pipf)
                        + 2*pipa*(pipf*(enga**2-malp**2) 
                                + mass**2*(malp**2-enga**2)+malp**2*engf*(enga+2*engi))
                        - 2*malp**4*engi*engf)
        term4 = 2*pipa*pfpa**3

        msquared = 4*factor*(term1+term2+term3-term4) / den1

        return msquared

    def _get_m2_zero(self,cif,cia,delta):

        enga     = self._alp.energy.value
        charge   = self._initial.charge
        ga_sm    = self._alp.e_coupling
        n_eff    = self._colapsar.n_eff_N.value
        temp     = self._colapsar.temperature.value
        mass     = self._initial.mass.value
        cfa      = self._get_cfa(cif,cia,delta)

        pipf     = self._four_momentum_dot(self._initial,self._final,cif)
        pipa     = self._four_momentum_dot(self._initial,self._alp,cia)
        pfpa     = self._four_momentum_dot(self._final,self._alp,cfa)
        transfer = self._p_transfer(cif,cia,delta)

        val1   = ga_sm**2 * charge**4 * n_eff * temp
        val2   = 2.*transfer**2 * (temp*transfer**2 + n_eff*charge**2)
        factor = val1/val2

        term1 = 2*enga**2*(pipf-mass**2-pfpa+pipa)/(pipa*pfpa)

        term2 = pfpa/pipa
        term3 = pipa/pfpa

        msquared = factor*(term1+2-term2-term3)

        return msquared
    
    def _get_alprho(self,cif,cia,delta):

        m2 = self._get_m2(cif,cia,delta)

        pi = self._initial.norm_spat_p.value
        pa = self._alp.norm_spat_p.value
        pf = self._final.norm_spat_p.value

        fd_engi  = self._get_fd_init()
        fd_engf  = self._get_pauliblock()
        values   = pi*pa*pf*m2*fd_engi*fd_engf/(64*np.pi**6)

        flux = self._deltar.value * values

        return flux

    def _get_alprho_zero(self,cif,cia,delta):

        m2 = self._get_m2_zero(cif,cia,delta)

        pi = self._initial.norm_spat_p.value
        pa = self._alp.norm_spat_p.value
        pf = self._final.norm_spat_p.value

        fd_engi  = self._get_fd_init()
        fd_engf  = self._get_pauliblock()
        values   = pi*pa*pf*m2*fd_engi*fd_engf/(64*np.pi**6)

        flux = self._deltar.value * values

        return flux

    def _dndedt_einterval(self,ninterval):

        np.random.seed(int(time.time()))

        integral   = 0

        thiserange = self._efintervals[ninterval]

        lowbounds  = [thiserange[0],self._cifrange[0],
                      self._ciarange[0],self._deltarange[0]]
        highbounds = [thiserange[1],self._cifrange[1],
                      self._ciarange[1],self._deltarange[1]]

        samples = np.random.uniform(low=lowbounds,high=highbounds,
                                    size=(self._nsteps,self._ndim))

        eprob      = thiserange[1]-thiserange[0]
        cifprob    = np.diff(self._cifrange)[0]
        ciaprob    = np.diff(self._ciarange)[0]
        deltaprob  = np.diff(self._deltarange)[0]
        total_prob = eprob*cifprob*ciaprob*deltaprob

        for sample in samples:
            engf,cif,cia,delta = sample

            self._final.energy   = engf*nat.MeV
            self._initial.energy = self._alp.energy + self._final.energy

            vfunc = self._get_alprho(cif,cia,delta)

            integral += vfunc

        MeV2tocm2inv   = nat.convert(nat.MeV**2,nat.cm**(-2))
        MeVtosecinv    = nat.convert(nat.MeV,nat.s**(-1))
        invMeVtoinvTeV = nat.convert(nat.MeV**(-1),nat.TeV**(-1))

        integral *= invMeVtoinvTeV*MeV2tocm2inv*MeVtosecinv

        integral /= self._nsteps
        integral *= total_prob

        return integral

    def _dndedt_einterval_zero(self,ninterval):

        np.random.seed(int(time.time()))

        integral   = 0

        thiserange = self._efintervals[ninterval]

        lowbounds  = [thiserange[0].value,self._cifrange[0],
                      self._ciarange[0],self._deltarange[0]]
        highbounds = [thiserange[1].value,self._cifrange[1],
                      self._ciarange[1],self._deltarange[1]]

        samples = np.random.uniform(low=lowbounds,high=highbounds,
                                    size=(self._nsteps,self._ndim))

        eprob      = thiserange[1].value-thiserange[0].value
        cifprob    = np.diff(self._cifrange)[0]
        ciaprob    = np.diff(self._ciarange)[0]
        deltaprob  = np.diff(self._deltarange)[0]
        total_prob = eprob*cifprob*ciaprob*deltaprob

        for sample in samples:
            engf,cif,cia,delta = sample

            self._final.energy   = engf*nat.MeV
            self._initial.energy = self._alp.energy + self._final.energy

            vfunc = self._get_alprho_zero(cif,cia,delta)

            integral += vfunc

        integral /= self._nsteps
        integral *= total_prob

        MeV2tocm2inv   = nat.convert(nat.MeV**2,nat.cm**(-2))
        MeVtosecinv    = nat.convert(nat.MeV,nat.s**(-1))
        invMeVtoinvTeV = nat.convert(nat.MeV**(-1),nat.TeV**(-1))

        integral *= invMeVtoinvTeV*MeV2tocm2inv*MeVtosecinv

        return integral

    def get_dndedt(self):
            # e_alp,malp,g_coupling,emin,emax,nintervals,
            #     ciabounds,cifbounds,deltabounds,log_prob,
            #     ndim=4,nwalkers=50,burnsteps=100,nsteps=1000,moves=emcee.moves.DEMove(),
            #     mass=0.511,charge=0.303,n_eff=2.e+5,
            #     temperature=30,chem_pot=200,delta_r=2.5e+16,npoints=1000):
        
        integral = 0

        for nint in range(self._nintervals):

            alp_spectra = self._dndedt_einterval(nint)
            
            if alp_spectra < self._tol:
                alp_spectra = self._tol

            integral += alp_spectra

        integral *= nat.TeV**(-1)*nat.s**(-1)*nat.cm**(-2)

        self._prod_rate = integral

        return integral

    def get_dndedt_zero(self):
            # e_alp,malp,g_coupling,emin,emax,nintervals,
            #     ciabounds,cifbounds,deltabounds,log_prob,
            #     ndim=4,nwalkers=50,burnsteps=100,nsteps=1000,moves=emcee.moves.DEMove(),
            #     mass=0.511,charge=0.303,n_eff=2.e+5,
            #     temperature=30,chem_pot=200,delta_r=2.5e+16,npoints=1000):
        
        integral = 0

        for nint in range(self._nintervals):

            alp_spectra = self._dndedt_einterval_zero(nint)
            
            if alp_spectra < self._tol:
                alp_spectra = self._tol

            integral += alp_spectra

        integral *= nat.TeV**(-1)*nat.s**(-1)*nat.cm**(-2)

        self._prod_rate = integral

        return integral
