# Load packages, including CAMB
import sys, platform, os
import matplotlib.pyplot as plt

plt.style.use('sanglier')

#from scipy import constants, special, integrate, stats
import numpy as np
from scipy.interpolate import CubicSpline, RegularGridInterpolator
from scipy import integrate
from hankel import HankelTransform

#Assume installed from github using "git clone --recursive https://github.com/cmbant/CAMB.git"
#This file is then in the docs folders
camb_path = os.path.realpath(os.path.join(os.getcwd(),'..'))
sys.path.insert(0,camb_path)
import camb
from camb import model, initialpower
from camb.correlations import cl2corr
print('Using CAMB %s installed at %s'%(camb.__version__,os.path.dirname(camb.__file__)))

# jax and disco-dj
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from discoeb.background import evolve_background
from discoeb.perturbations import evolve_perturbations, get_power

def radtoarcmin(angle_rad):
    """
    This function converts an an angle expressed in radians
    into arcmins.
    """
    
    angle_arcmin = angle_rad * 60 * 180 / np.pi
    
    return angle_arcmin


def arcmintorad(angle_arcmin):
    """
    This function converts an an angle expressed in arcmins
    into radians.
    """
    
    angle_rad = angle_arcmin / (60 * 180 / np.pi)
    
    return angle_rad

# set up consistent cosmology
Tcmb    = 2.7255
YHe     = 0.248
Omegam  = 0.3099
Omegab  = 0.0488911
Omegac  = Omegam - Omegab
w_DE_0  = -0.99
w_DE_a  = 0.0
cs2_DE  = 1.0
num_massive_neutrinos = 1
mnu     = 0.06  #eV
Tnu     = (4/11)**(1/3) #0.71611 # Tncdm of CLASS
Neff    = 3.046 # -1 if massive neutrino present
N_nu_mass = 1
N_nu_rel = Neff - N_nu_mass * (Tnu/((4/11)**(1/3)))**4
h       = 0.67742
A_s     = 2.1064e-09
n_s     = 0.96822
k_p     = 0.05

# modes to sample
nmodes = 10#512
kmin = 1e-4
kmax = 1e+1
aexp = 0.01

# Compute Background evolution
param = {}
param['Omegam']  = Omegam
param['Omegab']  = Omegab
# param['OmegaDE'] = OmegaDE
param['w_DE_0']  = w_DE_0
param['w_DE_a']  = w_DE_a
param['cs2_DE']  = cs2_DE
param['Omegak']  = 0.0
param['A_s']     = A_s
param['n_s']     = n_s
param['H0']      = 100*h
param['Tcmb']    = Tcmb
param['YHe']     = YHe
param['Neff']    = N_nu_rel
param['Nmnu']    = N_nu_mass
param['mnu']     = mnu
param['k_p'] = k_p

print('got params')

param = evolve_background(param=param, thermo_module='RECFAST')#, class_thermo=thermo)

print('disco bg evolved')

aexp_out = jnp.array([aexp])
y, kmodes = evolve_perturbations(param=param, kmin=kmin, kmax=kmax, num_k=nmodes, aexp_out=aexp_out, rtol=1e-3, atol=1e-3)

print('disco perturbations evolved')

Pkm = get_power(k=kmodes, y=y[:,0,:], idx=4, param=param)

print('disco PS computed')

Omeganu = param['Omegamnu']
g = (Tnu / (11/4)**(-1/3))**4 # 11/4 is g_0/g_*
print('setting camb pars')
pars = camb.CAMBparams(H0=100*h, ombh2=Omegab*h**2, omch2=Omegac*h**2, omnuh2=Omeganu*h**2, omk=0.0, YHe=YHe, TCMB=Tcmb, 
                        num_nu_massive=N_nu_mass, num_nu_massless=N_nu_rel, nu_mass_eigenstates = 1,
                        nu_mass_fractions=[1.], nu_mass_degeneracies=[g * N_nu_mass], nu_mass_numbers=[1], share_delta_neff=False, MassiveNuMethod='Nu_int' )
                        # nu_mass_fractions=[1.], nu_mass_degeneracies=[0.], nu_mass_numbers=[1], share_delta_neff=True, MassiveNuMethod='Nu_int' )
                        # nu_mass_numbers=[nnu], nu_mass_degeneracies=[Neff / nnu],
                        # nu_mass_fractions=[1.], share_delta_neff=True)
                       #num_nu_massive=num_massive_neutrinos, num_nu_massless=Neff)
#This function sets up with one massive neutrino and helium set using BBN consistency
# pars.set_cosmology(H0=100*h, ombh2=Omegab*h**2, omch2=(Omegam-Omegab)*h**2, mnu=mnu, omk=0., 
#                    tau=None, num_massive_neutrinos=num_massive_neutrinos, standard_neutrino_neff = Neff+num_massive_neutrinos,
#                    nnu=Neff+num_massive_neutrinos, YHe=YHe, TCMB=Tcmb )
pars.set_dark_energy(w=w_DE_0, cs2=cs2_DE, wa=w_DE_a, dark_energy_model='fluid')
pars.InitPower.set_params(As=A_s, ns=n_s, r=0)
pars.set_accuracy(AccuracyBoost=3.0 , DoLateRadTruncation=False, lAccuracyBoost=3.0)
pars.set_matter_power(redshifts=[1/aexp-1], kmax=kmax/h*1.1, accurate_massive_neutrino_transfers=True)
pars.Reion.Reionization = False
pars.Transfer.high_precision = True
pars.Transfer.accurate_massive_neutrinos = True
pars.MassiveNuMethod= 'Nu_int'
print('getting camb results')
results = camb.get_results(pars)

# # CAMB parameters
# pars = camb.CAMBparams()
# pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, TCMB=cmb_temp, omk=Omega_K)
# pars.InitPower.set_params(ns=ns)
# background = camb.get_background(pars)

# print('computing Weyl from CAMB')

# Compute Weyl power spectrum
# zmax = 7
# kmax = 5e2 #(inverse Mpc)
# extrap_kmax = 1e10
# Weyl_power_spectra = camb.get_matter_power_interpolator(pars, zmax=zmax, kmax=kmax, zs=None, hubble_units=False, k_hunit=False, var1=model.Transfer_Weyl, var2=model.Transfer_Weyl, extrap_kmax=extrap_kmax)

# matter_power_spectra = camb.get_matter_power_interpolator(pars, zmax=zmax, kmax=kmax)
print('getting camb PS')
kh_camb, z_camb, pkm_camb  = results.get_matter_power_spectrum(minkh=kmin/h*1.001, maxkh=kmax/h*0.999, npoints = nmodes, var1='delta_tot', var2='delta_tot')

print('plotting')
fig, ax = plt.subplots(1,1, figsize=(6,4))
ax.plot(kmodes, Pkm, lw=3, label = 'DISCO-DJ')
ax.loglog(kh_camb[1:]*h, pkm_camb[0,1:]/h**3, lw=3, label = 'CAMB')
ax.set_xlabel('$k$ [Mpc$^{-1}$]')
ax.set_ylabel(r'$P_{\rm m}(k)$ [Mpc$^{3}$]')
ax.legend()
plt.show()

exit()

# k = np.logspace(-4, 1, 100)

# pweyl = Weyl_power_spectra.P(0,k)

# print('done')

# np.savetxt('/home/nataliehogg/Documents/Projects/6x2pt/git_focsle/notebooks/camb_wps.txt', np.array((k, pweyl)))

# zplot = [0, 3]
# for z in zplot:
#     plt.loglog(k, Weyl_power_spectra.P(z,k), lw=3)
# plt.xlabel('$k$ (Mpc$^{-1}$)')
# plt.ylabel('$P_W$ (Mpc$^{-1}$)')
# plt.legend(['$z=%s$'%z for z in zplot], frameon=False);

# plt.show()

# print('computing P(k) from DISCO-DJ')

# @jax.jit
# def compute_matter_power(sigma8_fid):
#     param = {}
#     param['Omegam']  = 0.301            # Total matter density parameter
#     param['Omegab']  = Omega_b          # Baryon density parameter
#     param['w_DE_0']  = -0.99             # Dark energy equation of state parameter today
#     param['w_DE_a']  = 0.0               # Dark energy equation of state parameter time derivative
#     param['cs2_DE']  = 1.0               # Dark energy sound speed squared
#     param['Omegak']  = Omega_K               # Omega_DE computed directly thanks to this assumption
#     param['A_s']     = 2.1064e-09        # Scalar amplitude of the primordial power spectrum
#     param['n_s']     = ns          # Scalar spectral index
#     param['H0']      = H0            # Hubble constant today in units of 100 km/s/Mpc
#     param['Tcmb']    = cmb_temp           # CMB temperature today in K
#     param['YHe']     = 0.248             # Helium mass fraction
#     param['Neff']    = 2.046             # Effective number of ultrarelativistic neutrinos; -1 for massive neutrinos
#     param['Nmnu']    = 1                 # Number of massive neutrinos (must be 1 currently)
#     param['mnu']     = 0.06              # Sum of neutrino masses in eV 
#     param['k_p']     = 0.05              # Pivot scale in 1/Mpc
    
# #     param['sigma8'] = sigma8_fid
    
#     # modes to sample
#     nmodes = 512                         # 512; number of modes to sample
#     kmin   = 1e-4                        # 1e-4; minimum k in 1/Mpc
#     kmax   = 10                           # 10; maximum k in 1/Mpc
#     aexp   = 1.0                         # scale factor at which to evaluate the power spectrum

#     print('got params')
  
#     # Compute Background+thermal evolution
#     param = evolve_background(param=param, thermo_module='RECFAST')

#     print('evolved background')
    
#     # compute perturbation evolution
#     aexp_out = jnp.array([aexp])
#     y, kmodes = evolve_perturbations(param=param, kmin=kmin, kmax=kmax, 
#                                      num_k=nmodes, aexp_out=aexp_out, 
#                                      rtol=1e-3, atol=1e-3)

#     print('evolved perturbations')
    
#     # turn perturbations into power spectra
#     Pkm = get_power(k=kmodes, y=y[:,0,:], idx=4, param=param)

#     print('got power')
    
#     return Pkm, kmodes

# Pkm, kmodes = compute_matter_power(0.8)

# np.savetxt('/home/nataliehogg/Documents/Projects/6x2pt/git_focsle/notebooks/disco_mps.txt', np.array((kmodes, Pkm)))

# compute_PS_jit = jax.jit(compute_matter_power)

# # Pre-compile the function before timing...
# compute_PS_jit(sigma8_fid).block_until_ready()

# Pkm, kmodes = compute_PS_jit(0.8)

print('got P(k)')

# print(Pkm)

print('plotting')

# compare CAMB and DISCO-DJ output

fig, ax = plt.subplots(1,1, figsize=(6,4))
ax.plot(kmodes, Pkm, lw=3, label = 'DISCO-DJ')
ax.loglog(k, matter_power_spectra.P(0,k), lw=3, label = 'CAMB')
ax.set_xlabel('$k$ (Mpc$^{-1}$)')
ax.set_ylabel(r'$P_{\rm m}(k)$ (Mpc$^{-1}$)')
ax.legend()
plt.show()

exit()

@jax.jit
def matter_to_weyl(pm, H0, Omega_m, z_fiducial):
    '''
    take a DISCO-DJ matter power spectrum and return the Weyl power spectrum
    '''
    clight = 3e5 # km/s
    prefactor = (((3/2)*(H0**2)*Omega_m)**2)*((1+z_fiducial)**2)
    pw = pm/prefactor

    return pw/(clight**2)

Pk_weyl = matter_to_weyl(Pkm, 67, 0.3, 0)


fig, ax = plt.subplots(1,1, figsize=(6,4))
ax.plot(kmodes, Pk_weyl*kmodes**2, lw=3, label = 'DISCO-DJ')
ax.loglog(k, Weyl_power_spectra.P(0,k), lw=3, label = 'CAMB')
ax.set_xlabel('$k$ (Mpc$^{-1}$)')
ax.set_ylabel(r'$P_{\rm W}$ (Mpc$^{-1}$)')
ax.legend()
plt.show()