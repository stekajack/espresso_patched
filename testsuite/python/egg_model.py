import numpy as np

import unittest as ut
import unittest_decorators as utx

import espressomd
from espressomd.observables import MagneticDipoleMoment
from espressomd.virtual_sites import VirtualSitesRelative

import scipy.special as spcl
from scipy.integrate import quad

def random_unit_vector():

    rand1 = np.random.random()
    rand2 = np.random.random()
    cos_theta = -1.0 + 2 * rand1
    sin_theta = (1 - cos_theta * cos_theta)**.5
    cos_phi = np.cos(2 * np.pi * rand2)
    sin_phi = np.sin(2 * np.pi * rand2)

    return np.array([cos_theta * cos_phi, cos_theta * sin_phi, sin_theta])

langevin_func = lambda xi: 1./np.tanh(xi) - 1./xi

def magn_fixed_ea(xi,sigma,psi):
  z_int = lambda x,xi,s,p: np.exp(s*x**2) * np.cosh(xi*x*np.cos(p)) * spcl.i0(xi*(1-x**2)**0.5*np.sin(p))
  dz_int = lambda x,xi,s,p: np.exp(s*x**2) * (
          np.cosh(xi*x*np.cos(p)) * spcl.i1(xi*(1-x**2)**0.5*np.sin(p)) * (1-x**2)**0.5 * np.sin(p) +
          np.sinh(xi*x*np.cos(p)) * spcl.i0(xi*(1-x**2)**0.5*np.sin(p)) * x * np.cos(p)
          )
  part_func = lambda xi,s,p: quad(lambda x: z_int(x,xi,s,p),0,1)[0]
  d_part_func = lambda xi,s,p: quad(lambda x: dz_int(x,xi,s,p),0,1)[0]

  return d_part_func(xi,sigma,psi)/part_func(xi,sigma,psi)

tau_par = lambda tau_d, sigma: tau_d * (np.exp(sigma) - 1)/(2  * sigma)/(1/(1 + 1/sigma)*(sigma/np.pi)**0.5 + 2**(-sigma - 1)) if (sigma > 0) else tau_d
order_param = lambda s: 0.75 * (s**0.5/spcl.dawsn(s**0.5) - 1)/s - 0.5
tau_per = lambda tau_d, sigma: 2 * tau_d * (1 - order_param(sigma) )/(2 + order_param(sigma))

tau_par_liq = lambda tau_d, sigma, tau_b: 1./(1./tau_b + 1./tau_par(tau_d,sigma))
tau_per_liq = lambda tau_d, sigma, tau_b: 1./(1./tau_b + 1./tau_per(tau_d,sigma))

acorr_par = lambda t, td, s: np.exp(-t/tau_par(td,s))
acorr_per = lambda t, td, s: np.exp(-t/tau_per(td,s))

acorr_par_liq = lambda t, td, s, tb: np.exp(-t/tau_par_liq(td,s,tb))
acorr_per_liq = lambda t, td, s,tb: np.exp(-t/tau_per_liq(td,s,tb))

acorr_rand = lambda t,td,sigma: acorr_par(t,td,sigma)*(1+2*order_param(sigma))/3 + 2*acorr_per(t,td,sigma)*(1-order_param(sigma))/3
acorr_liq = lambda t,td,sigma,tb: acorr_par_liq(t,td,sigma,tb)*(1+2*order_param(sigma))/3 + 2*acorr_per_liq(t,td,sigma,tb)*(1-order_param(sigma))/3

        
@utx.skipIfMissingFeatures(["EGG_MODEL"])
class EggModelTest(ut.TestCase):

    num_particles = 100
    magcurve_tol = 0.01
    magcurve_warmup_nsteps = 1000
    magcurve_nsteps = 170000
    
    acf_tol = 0.015
    acf_nsteps = 500000
    acf_max_time = 1
    acf_warmup_steps = 1000
    
    fields = [1,3]
    egg_gamma = 1
    gamma = 1
    sigma = 4
    texture_angle = np.pi/2.5
    
    tau_d = egg_gamma/2.
    tau_b = gamma/2.
    
    box_length = 10
    system = espressomd.System(box_l=box_length*np.ones(3))
    system.periodicity = [1,1,1]
    system.time_step = 0.005
    system.cell_system.skin = 0.2
    system.min_global_cut = 1
        
    system.virtual_sites = VirtualSitesRelative(have_quaternion=True)

    
    def generate_ensemble(self, fixed, psi = 0):
        
        system = self.system 
        
        for i in range(self.num_particles):
            r_id = len(system.part)
            v_id = r_id + 1
            pos = self.box_length  * np.random.random(3)
            if fixed:
                system.part.add(pos=pos, type = 0, id = r_id, rotation=[0,0,0], director = [0,0,1], dipm = 0, fix = [True,True,True])
            else:
                system.part.add(pos=pos, type = 0, id = r_id, rotation=[1,1,1], director = [0,0,1], dipm = 0, fix = [True,True,True])
                
            system.part.add(pos=pos, type = 1, id = v_id, rotation=[1,1,1], dip = random_unit_vector(), virtual = True)
            system.part.by_id(v_id).vs_auto_relate_to(r_id)

            system.part.by_id(v_id).egg_model_params = {"use_egg_model": True, "egg_gamma": self.egg_gamma, "aniso_energy": self.sigma} 
            system.part.by_id(v_id).vs_quat = system.part.by_id(v_id).quat
            system.part.by_id(v_id).axis_quat_body  = (np.cos(psi/2),np.sin(psi/2),0,0)
            
    def calc_magcurve(self):
        
        system = self.system
        
        m_ids = system.part.select(type=1).id
        dipm_tot = MagneticDipoleMoment(ids=m_ids)
        
        magnetic_field = espressomd.constraints.HomogeneousMagneticField(H=[0,0,0])
        self.system.constraints.add(magnetic_field)
        magcurve = []
        for xi in self.fields:

            magnetic_field.H = [0,0,xi]   
            
            system.integrator.run(self.magcurve_warmup_nsteps)
            
            mmoment_acc = espressomd.accumulators.TimeSeries(obs=dipm_tot)
            system.auto_update_accumulators.add(mmoment_acc)
            
            system.integrator.run(self.magcurve_nsteps)
            mz = np.mean(mmoment_acc.time_series()[:,2])/self.num_particles
            magcurve.append(mz)
            mmoment_acc.clear()
                    
        return magcurve
            
            
    def calc_acf(self):
            
        system = self.system
        
        m_ids = system.part.select(type=1).id
        dipm_tot = MagneticDipoleMoment(ids=m_ids)
        
        system.integrator.run(self.acf_warmup_steps)
        
        magm_corr = espressomd.accumulators.Correlator(obs1=dipm_tot, tau_lin=16, tau_max=100, delta_N=1, corr_operation="scalar_product", compress1="discard1")
        system.auto_update_accumulators.add(magm_corr)
        
        system.integrator.run(self.acf_nsteps)
    
        magm_corr.finalize()
        
        ind = np.where(magm_corr.lag_times() > self.acf_max_time)[0][0]
        times = (magm_corr.lag_times())[:ind]
        acf = magm_corr.result()
        acf = (acf/acf[0])[:ind,0]
        
        return times, acf

    def tearDown(self):
        self.system.part.clear()
        self.system.thermostat.turn_off()
        self.system.auto_update_accumulators.clear()
        self.system.constraints.clear()
        
    def setUp(self):
        
        np.random.seed(18)
        self.system.thermostat.set_brownian(kT=1., gamma=self.gamma, seed=18, act_on_virtual = True)
        self.system.integrator.set_brownian_dynamics()
        
        
    def test_magnetic_acf_fixed(self):
        
        self.generate_ensemble(fixed = True)
        times, sim_acf = self.calc_acf()
        theor_acf = acorr_rand(times,self.tau_d,self.sigma)
        np.testing.assert_allclose(sim_acf, theor_acf, atol=self.acf_tol)

    def test_magnetic_acf_liquid(self):
        
        self.generate_ensemble(fixed = False)
        times, sim_acf = self.calc_acf()
        theor_acf = acorr_liq(times,self.tau_d,self.sigma,self.tau_b)
        np.testing.assert_allclose(sim_acf, theor_acf, atol=self.acf_tol)
        
    def test_magcurve_fixed_parallel(self):
        
        self.generate_ensemble(fixed = True)
        sim_magcurve = self.calc_magcurve()
        theor_magcurve = np.array([magn_fixed_ea(xi,self.sigma,0) for xi in self.fields])
        np.testing.assert_allclose(sim_magcurve, theor_magcurve, atol=self.magcurve_tol)
        
    def test_magcurve_fixed_nonparallel(self):
        
        self.generate_ensemble(fixed = True, psi = self.texture_angle)
        sim_magcurve = self.calc_magcurve()
        theor_magcurve = np.array([magn_fixed_ea(xi,self.sigma,self.texture_angle) for xi in self.fields])
        np.testing.assert_allclose(sim_magcurve, theor_magcurve, atol=self.magcurve_tol)
        
    def test_magcurve_liquid(self):

        self.generate_ensemble(fixed = False)
        sim_magcurve = self.calc_magcurve()
        theor_magcurve = np.array([langevin_func(xi) for xi in self.fields])
        np.testing.assert_allclose(sim_magcurve, theor_magcurve, atol=self.magcurve_tol)

if __name__ == '__main__':
    ut.main()
