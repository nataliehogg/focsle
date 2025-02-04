# Load packages, including CAMB
import sys, platform, os
import matplotlib.pyplot as plt

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from discoeb.background import evolve_background
from discoeb.perturbations import evolve_perturbations, get_power

plt.style.use('sanglier')

colors = ['#1b9e77','#d95f02','#7570b3','#e7298a']

#from scipy import constants, special, integrate, stats
import numpy as np

camb_power_array = np.genfromtxt('/home/nataliehogg/Documents/Projects/6x2pt/git_focsle/notebooks/camb_wps.txt')

k = camb_power_array[0]
camb_weyl = camb_power_array[1]

disco_power_array = np.genfromtxt('/home/nataliehogg/Documents/Projects/6x2pt/git_focsle/notebooks/disco_mps.txt')

kmodes = disco_power_array[0]
Pkm = disco_power_array[1]


@jax.jit
def matter_to_weyl(pm, H0, Omega_m, z_fiducial):
    '''
    take a DISCO-DJ matter power spectrum and return the Weyl power spectrum
    '''
    clight = 3e5 # km/s
    prefactor = (((3/2)*(H0**2)*Omega_m)**2)*((1+z_fiducial)**2)
    pw = pm/prefactor

    return pw

Pk_weyl = matter_to_weyl(Pkm, 67, 0.3, 0)


fig, ax = plt.subplots(1,1, figsize=(6,4))
ax.plot(kmodes, Pk_weyl, lw=3, label = 'DISCO-DJ',  color=colors[0])
ax.plot(kmodes, Pk_weyl/3e5, lw=3, label = 'DISCO-DJ/c',  color=colors[1])
ax.plot(kmodes, Pk_weyl/3e5**2, lw=3, label = r'DISCO-DJ/c**2',  color=colors[2])

ax.loglog(k, camb_weyl, lw=3, label = 'CAMB', color='k')
ax.set_xlabel('$k$ (Mpc$^{-1}$)')
ax.set_ylabel(r'$P_{\rm W}$ (Mpc$^{-1}$)')
ax.legend()
plt.show()