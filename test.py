import sys
import os
import numpy as np
import scipy.integrate as integrate
import camb
from matplotlib import pyplot as plt
from scipy.stats import binned_statistic
import time
from camb import model
import pandas as pd
from astropy import constants as const
import scipy.interpolate as interpolate


def a(x):
    return x**3

f = lambda x, y, z : a(x) + y**2 +z**2

# instalacion de camb
camb_path = os.path.realpath(os.path.join(os.getcwd(), '..'))
sys.path.insert(0, camb_path)


print('Using CAMB %s installed at %s'%(camb.__version__, os.path.dirname(camb.__file__)))


# Creación de funciones E(z) y D(z), a manera de prueba se ocuparan los
# valores de Planck 2016 para testear las funciones

params_P18 = dict()
# crearemos diccionarios en donde estaran los parametros cosmologicos que
# queremos utilizar, es facil poder crear y modificar diccionarios

params_P18['Ob'] = 0.05   # Omega_b_0
params_P18['Om'] = 0.32  # Omega_m_0
params_P18['ns'] = 0.96605       # indice espectral
params_P18['ODE'] = 0.68   # Omega_DE_0
params_P18['sigma8'] = 0.816    # amplitud de densidades de fluctuación
params_P18['H0'] = 67.32      # 100h
params_P18['sum_mv'] = 0.06  # valor de masas de neutrino
params_P18['w_0'] = -1
params_P18['w_a'] = 0
params_P18['gamma'] = 0.55
params_P18['Ov'] = 0  # en este caso tomamos la densidad de radiación como nula

# Creacion de parametros con CAMB

pars = camb.CAMBparams()

# This function sets up CosmoMC-like settings, with one massive neutrino and
# helium set using BBN consistency

pars.set_dark_energy(w=-1.0, wa=0, dark_energy_model='fluid')
pars.set_cosmology(H0=67.4, ombh2=0.02233, omch2=0.1198, omk=0, tau=0.054)
results = camb.get_results(pars)

params_CAMB = dict()
# crearemos diccionarios en donde estaran los parametros cosmologicos que
# queremos utilizar, es facil poder crear y modificar diccionarios

params_CAMB['Ob'] = results.get_Omega('baryon')
params_CAMB['Om'] = 1 - pars.omk - results.get_Omega('de')
params_CAMB['ODE'] = results.get_Omega('de')
params_CAMB['H0'] = pars.H0
params_CAMB['w_0'] = pars.DarkEnergy.w
params_CAMB['w_a'] = pars.DarkEnergy.wa
params_CAMB['Ov'] = results.get_Omega('photon')

# cration of basic background quantities


def Omega_Lambda(Omega_m):
    """La funcion Omega_Lambda nos entregara este parametro en base a los que
    tenemos, para esto tambien debemos calcular Omega c, un parametro que
    no se utilizara, por lo que no es necesario almacenar"""
    return 1 - Omega_m


def Omega_K_0(Omega_DE, Omega_m):
    """Omega_K_0 nos entrega este parametro que es depende de Omega DE y
    Omega m, en el caso de del modelo ΛCDM este valor es cero"""
    return 1 - (Omega_DE + Omega_m)


def cosmological_parameters(cosmo_pars=dict()):
    """cosmological_parameters extrae los parametros necesarios para las
    el calculo de funciones E(z) y D(z), concadena estos parametros de manera
    que sean facil de utilizar, el default son los parametros de Planck 2018"""
    H0 = cosmo_pars.get('H0', params_CAMB['H0'])
    Om = cosmo_pars.get('Om', params_CAMB['Om'])
    Ob = cosmo_pars.get('Ob', params_CAMB['Ob'])
    Ov = cosmo_pars.get('Ov', params_CAMB['Ov'])
    ODE = cosmo_pars.get('ODE', params_CAMB['ODE'])
    OL = Omega_Lambda(Om)
    OK = Omega_K_0(ODE, Om)
    wa = cosmo_pars.get('wa', params_CAMB['w_a'])
    w0 = cosmo_pars.get('w0', params_CAMB['w_0'])
    return H0, Om, ODE, OL, OK, wa, w0


def E_arb(z, cosmo_pars=dict()):
    """E_arb es la función E(z) arbitraria para cualquier modelo cosmologico"""
    H0, Om, ODE, OL, Ok, wa, w0 = cosmological_parameters(cosmo_pars)
    exp = np.exp(-3*wa*(z/1+z))
    ind = 1 + wa + w0
    return np.sqrt(Om*(1+z)**3 + ODE*((1+z)**(3*ind))*exp + Ok*(1+z)**2)


def E(z, cosmo_pars=dict()):
    """E es la función E(z) para el caso w0 = -1 y wa = 0"""
    H0, Om, ODE, OL, Ok, wa, w0 = cosmological_parameters(cosmo_pars)
    return np.sqrt(Om*(1+z)**3 + OL + Ok*(1+z)**2)


# comoving distance to an object redshift z

def f_integral(z, cosmo_pars=dict()):
    """f_integral define la funcion dentro de la integral
    ocupada para el calculo de r(z)"""
    return 1/E(z, cosmo_pars)


def r(z, cosmo_pars=dict()):
    """r calcula comoving distnace to an objecto redshift"""
    H0, Om, ODE, OL, Ok, wa, w0 = cosmological_parameters(cosmo_pars)
    c = const.c.value / 1000 
    cte = c/H0
    f = lambda z: 1/ E(z, cosmo_pars)
    z_int = np.linspace(0, z, 200)
    return cte*np.trapz(f(z_int), z_int)

z_arr = np.linspace(0, 2.5, 100)

# fig, ax = plt.subplots(1, 1, sharey='row', sharex='col', figsize=(10, 8))
# for z in z_arr:
#     ax.scatter(z, r(z), s=1.0, label='$r(z)$', color='mediumpurple')
# ax.set_xlabel('Redshift $z$')
# ax.set_ylabel('$Comoving distance r(z)$')
# ax.set_title('Comoving distance $r(z)$ as a function of redshift $z$')
# fig.show()

# Bin creation

z_bin = binned_statistic(z_arr, z_arr, bins=100)
z_bin_equi = binned_statistic(z_arr, z_arr, bins=10)
limits = [z_bin.bin_edges[0], z_bin.bin_edges[-1]]
z_equi = [(0.001, 0.42), (0.42, 0.56), (0.56, 0.68), (0.68, 0.79), (0.79, 0.90),
                    (0.90, 1.02), (1.02, 1.15) ,(1.15, 1.32), (1.32, 1.58), (1.58, 2.50)] # This are the values from the paper

# Parameters adopted to describe the photometric redshift distribution source

PRD = dict()

PRD['cb'] = 1.0
PRD['zb'] = 0.0
PRD['sigmab'] = 0.05
PRD['co'] = 1.0
PRD['zo'] = 0.1
PRD['sigmao'] = 0.05
PRD['fout'] = 0.1

# Tilde function of comoving distance


def tilde_r(z, cosmo_pars=dict()):
    c = const.c.value / 1000
    H0, Om, ODE, OL, Ok, wa, w0 = cosmological_parameters(cosmo_pars)
    cte = c/H0
    return r(z, cosmo_pars)/cte

# Photometric redshift estimates:


def n(z):
    zm = 0.9  # median redshift, value given by Euclid Red Book
    z0 = zm/(np.sqrt(2))
    frac = z/z0
    return (frac**2)*np.exp(-(frac)**(3/2))


# Photometric redshift ditribution of sources

def P_ph(zp, z):
    cb = PRD['cb']
    zb = PRD['zb']
    sigmab = PRD['sigmab']
    co = PRD['co']
    zo = PRD['zo']
    sigmao = PRD['sigmao']
    fout = PRD['fout']

    frac_1 = (1-fout)/(np.sqrt(2*np.pi)*sigmab*(1+z))

    frac_2 = fout/(np.sqrt(2*np.pi)*sigmao*(1+z))

    exp_1 = np.exp((-1/2)*((z-cb*zp-zb)/sigmab*(1+z))**2)

    exp_2 = np.exp((-1/2)*((z-co*zp-zo)/sigmao*(1+z))**2)

    return frac_1*exp_1 + frac_2*exp_2


# Defining integrals for photometric redshift estimates


def int_1(zp, z):
    return n(z)*P_ph(zp, z)




def n_i(z, i):
    ith_bin = z_equi[i]
    zi_l, zi_u = ith_bin
    z_int1 = np.linspace(zi_l, zi_u, 200)
    z_int2 = np.linspace(limits[0], limits[1], 200)
    X, Y = np.meshgrid(z_int1, z_int2)
    list1 = int_1(X, Y)
    I1 = np.trapz(int_1(z_int1, z), z_int1)
    I2 = np.trapz(np.trapz(list1, z_int2, axis=0),z_int1, axis=0)

    return I1/I2




# Matter Power spectrum following CAMB demo
pars = camb.CAMBparams()
pars.set_cosmology(H0=67.5, ombh2=0.02233, omch2=0.1198, omk=0, tau=0.054)
# Now get matter power spectra and sigma8 at redshift 0 and 0.8
pars.InitPower.set_params(ns=0.9652)
# Note non-linear corrections couples to smaller scales than you want
pars.set_matter_power(redshifts=z_bin[0], kmax=2.0)

# Linear spectra
pars.NonLinear = model.NonLinear_none
results = camb.get_results(pars)
kh, z, pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=1,
                                              npoints=200)
s8 = np.array(results.get_sigma8())

# Non-Linear spectra (Halofit)
pars.NonLinear = model.NonLinear_both
results.calc_power_spectra(pars)
kh_nonlin, z_nonlin, pk_nonlin = results.get_matter_power_spectrum(minkh=1e-4,
                                                                   maxkh=1,
                                                                   npoints=200)



# Storage power matter parameters

list_of_PMS = list(zip(kh, z, pk, kh_nonlin, z_nonlin, pk_nonlin))

lst_n_i = dict()

lst_n_i["bin_0"] = np.loadtxt("Bin_number_d_0.txt")
lst_n_i["bin_1"] = np.loadtxt("Bin_number_d_1.txt")
lst_n_i["bin_2"] = np.loadtxt("Bin_number_d_2.txt")
lst_n_i["bin_3"] = np.loadtxt("Bin_number_d_3.txt")
lst_n_i["bin_4"] = np.loadtxt("Bin_number_d_4.txt")
lst_n_i["bin_5"] = np.loadtxt("Bin_number_d_5.txt")
lst_n_i["bin_6"] = np.loadtxt("Bin_number_d_6.txt")
lst_n_i["bin_7"] = np.loadtxt("Bin_number_d_7.txt")
lst_n_i["bin_8"] = np.loadtxt("Bin_number_d_8.txt")
lst_n_i["bin_9"] = np.loadtxt("Bin_number_d_9.txt")

interpolate_n_i = dict()

interpolate_n_i["I_0"] = interpolate.interp1d(lst_n_i["bin_0"][:, 0],
                                              lst_n_i["bin_0"][:, 1])
interpolate_n_i["I_1"] = interpolate.interp1d(lst_n_i["bin_1"][:, 0],
                                              lst_n_i["bin_1"][:, 1])                                              
interpolate_n_i["I_2"] = interpolate.interp1d(lst_n_i["bin_2"][:, 0],
                                              lst_n_i["bin_2"][:, 1])
interpolate_n_i["I_3"] = interpolate.interp1d(lst_n_i["bin_3"][:, 0],
                                              lst_n_i["bin_3"][:, 1])
interpolate_n_i["I_4"] = interpolate.interp1d(lst_n_i["bin_4"][:, 0],
                                              lst_n_i["bin_4"][:, 1])
interpolate_n_i["I_5"] = interpolate.interp1d(lst_n_i["bin_5"][:, 0],
                                              lst_n_i["bin_5"][:, 1])
interpolate_n_i["I_6"] = interpolate.interp1d(lst_n_i["bin_6"][:, 0],
                                              lst_n_i["bin_6"][:, 1])
interpolate_n_i["I_7"] = interpolate.interp1d(lst_n_i["bin_7"][:, 0],
                                              lst_n_i["bin_7"][:, 1])
interpolate_n_i["I_8"] = interpolate.interp1d(lst_n_i["bin_8"][:, 0],
                                              lst_n_i["bin_8"][:, 1])
interpolate_n_i["I_9"] = interpolate.interp1d(lst_n_i["bin_9"][:, 0],
                                              lst_n_i["bin_9"][:, 1])


def W_int(z_1, z, i):
    return interpolate_n_i['I_%s'%(str(i))](z_1)*(1-tilde_r(z)/tilde_r(z_1))


def Window_F(z, i):
    z_int = np.linspace(z, limits[1], 200)
    return np.trapz(W_int(z_int, z, i), z_int)

start = time.time()
fig, ax = plt.subplots(1, 1, sharey='row', sharex='col', figsize=(10, 8))
for z in z_arr:
    ax.scatter(z, Window_F(z, 1), s=2.0, label='$Window Function(z)$',
               color='mediumpurple')
ax.set_xlabel('Redshift $z$')
ax.set_ylabel('Window Function $\tilde{W}_{1}(z)$')
ax.set_title('Window function for an specific bin $\tilde{W}(z)$ as a function of redshift $z$')
end = time.time()

print("El tiempo que se demoró es "+str(end-start)+" segundos")
fig.show()

def Convergence(l, i, j, z, cosmo_pars=dict()):
    H0, Om, ODE, OL, Ok, wa, w0 = cosmological_parameters(cosmo_pars)
    c = const.c.value / 1000
    cte = (c/H0)
    E = lambda z: np.sqrt(Om*(1+z)**3 + OL + Ok*(1+z)**2)
    r = lambda z: 1/ E(z)
    
    return 