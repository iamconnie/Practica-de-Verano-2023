# importancion de paquetes necesarios para el código

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
from scipy.stats import linregress

plt.rc('font', size=15)  #fontsize for plots
plt.rc('axes', titlesize=16)#fontsize of the title

# instalacion de camb

camb_path = os.path.realpath(os.path.join(os.getcwd(), '..'))
sys.path.insert(0, camb_path)


print('Using CAMB %s installed at %s' % (
      camb.__version__, os.path.dirname(camb.__file__)))


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
    if type(z) == np.ndarray:
        integral = np.zeros(200)
        for idx, redshift in enumerate(z):
            z_int = np.linspace(0, redshift, 200)
            integral[idx] = np.trapz(f_integral(z_int, cosmo_pars), z_int)
    else:
        z_int = np.linspace(0, z, 200)
        integral = np.trapz(f_integral(z_int, cosmo_pars), z_int)
    return const.c.value / 1000 / pars.H0 * integral


# transverse comoving distance


def D(z, cosmo_pars=dict()):
    """La funcion D calcula transverse comoving distance para los distintos
    casos de el parametro Omgea_K_0"""
    c = const.c.value / 1000
    H0, Om, ODE, OL, Ok, wa, w0 = cosmological_parameters(cosmo_pars)
    cte_1 = c/H0
    cte_2 = H0/c
    a = 1/(1+z)
    if Ok < 0:
        return a*(cte_1*(1/(np.abs(Ok)**(1/2))))*np.sin(
            np.abs(Ok)**(1/2)*cte_2*r(z, cosmo_pars))
    if Ok == 0:
        return a*r(z, cosmo_pars)
    if Ok > 0:
        return a*(cte_1*(1/(Ok**(1/2))))*np.sinh(
            (Ok**(1/2))*cte_2*r(z, cosmo_pars))
    else:
        return "Error"


# all plots in the same row, share the y-axis.

z_arr = np.linspace(0, 2.5, 100)
# fig, ax = plt.subplots(1, 1, sharey='row', sharex='col', figsize=(10, 8))

# Proper distance dependent on redshift plot
# ax.plot(z_arr, E_arb(z_arr), label='$E(z)$', color='mediumpurple')
# ax.set_xlabel('Redshift $z$')
# ax.set_ylabel('$E(z)$')
# ax.set_title('Proper distance $E(z) as a function of redshift $z$')
# plt.show()

# Comoving distance to an object redshift z plot

# fig, ax = plt.subplots(1, 1, sharey='row', sharex='col', figsize=(10, 8))
# for z in z_arr:
#     ax.scatter(z, r(z), s=1.0, label='$r(z)$', color='mediumpurple')
# ax.set_xlabel('Redshift $z$')
# ax.set_ylabel('$Comoving distance r(z)$')
# ax.set_title('Comoving distance $r(z)$ as a function of redshift $z$')
# plt.show()

# Angular diameter distance to an object redshift z plot

# fig, ax = plt.subplots(1, 1, sharey='row', sharex='col', figsize=(10, 8))
# for z in z_arr:
#     ax.scatter(z, D(z), s=1.0, label='$D_A(z)$', color='mediumpurple')
# ax.set_xlabel('Redshift $z$')
# ax.set_ylabel('$D_A(z)$')
# ax.set_title('Angular diameter distance $D_a(z)$ as a function of redshift $z$')
# plt.show()

# now using CAMB parameters

# Proper distance dependent on redshift plot
# fig, ax = plt.subplots(1, 1, sharey='row', sharex='col', figsize=(10, 8))
# ax.plot(z_arr, E_arb(z_arr, params_CAMB), label='$E(z)$', color='mediumpurple')
# ax.set_xlabel('Redshift $z$')
# ax.set_ylabel('$E(z)$')
# ax.set_title('Proper distance $E(z) as a function of redshift $z$')
# plt.show()

# Comoving distance to an object redshift z plot

# fig, ax = plt.subplots(1, 1, sharey='row', sharex='col', figsize=(10, 8))
# for z in z_arr:
#     ax.scatter(z, r(z, params_CAMB), s=1.0, label='$r(z)$',
#                color='mediumpurple')
# ax.set_xlabel('Redshift $z$')
# ax.set_ylabel('$Comoving distance r(z)$')
# ax.set_title('Comoving distance $r(z)$ as a function of redshift $z$')
# plt.show()

# Angular diameter distance to an object redshift z plot

# fig, ax = plt.subplots(1, 1, sharey='row', sharex='col', figsize=(10, 8))
# for z in z_arr:
#     ax.scatter(z, D(z, params_CAMB), s=1.0, label='$D_A(z)$',
#                color='mediumpurple')
# ax.set_xlabel('Redshift $z$')
# ax.set_ylabel('$D_A(z)$')
# ax.set_title('Angular diameter distance $D_a(z)$ as a function of redshift $z$')
# plt.show()


# Window Function

# Bin creation

z_bin = binned_statistic(z_arr, z_arr, bins=100)
z_bin_equi = binned_statistic(z_arr, z_arr, bins=10)
limits = [z_bin.bin_edges[0], z_bin.bin_edges[-1]]
# This are the values from the paper
z_equi = [(0.001, 0.42), (0.42, 0.56), (0.56, 0.68), (0.68, 0.79),
          (0.79, 0.90), (0.90, 1.02), (1.02, 1.15), (1.15, 1.32),
          (1.32, 1.58), (1.58, 2.50)]

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

# Converting lists of tuples into
# pandas Dataframe.
# df = pd.DataFrame(list_of_PMS,
#                   columns=['kh', 'z', 'pk', 'nonlinear_kh',
#                            'nonlinear_z', 'nonlinear_pk'])

# # Print data.
# df.to_csv('PMS_params.txt', sep='\t')

# Storage number density

z_list = z_bin[1]

# lst_0 = []
# for z in z_list:
#     lst_0.append([z, n_i(z, 0)])
# np.savetxt('Bin_number_d_0.txt', np.array(lst_0))


# lst_1 = []
# for z in z_list:
#     lst_1.append([z, n_i(z, 1)])
# np.savetxt('Bin_number_d_1.txt', np.array(lst_1))

# lst_2 = []
# for z in z_list:
#     lst_2.append([z, n_i(z, 2)])
# np.savetxt('Bin_number_d_2.txt', np.array(lst_2))

# lst_3 = []
# for z in z_list:
#     lst_3.append([z, n_i(z, 3)])
# np.savetxt('Bin_number_d_3.txt', np.array(lst_3))

# lst_4 = []
# for z in z_list:
#     lst_4.append([z, n_i(z, 4)])
# np.savetxt('Bin_number_d_4.txt', np.array(lst_4))

# lst_5 = []
# for z in z_list:
#     lst_5.append([z, n_i(z, 5)])
# np.savetxt('Bin_number_d_5.txt', np.array(lst_5))

# lst_6 = []
# for z in z_list:
#     lst_6.append([z, n_i(z, 6)])
# np.savetxt('Bin_number_d_6.txt', np.array(lst_6))

# lst_7 = []
# for z in z_list:
#     lst_7.append([z, n_i(z, 7)])
# np.savetxt('Bin_number_d_7.txt', np.array(lst_7))

# lst_8 = []
# for z in z_list:
#     lst_8.append([z, n_i(z, 8)])
# np.savetxt('Bin_number_d_8.txt', np.array(lst_8))

# lst_9 = []
# for z in z_list:
#     lst_9.append([z, n_i(z, 9)])
# np.savetxt('Bin_number_d_9.txt', np.array(lst_9))

# Dictionary of Bin number density

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


# Dictionary of Inerpolation for bin number density

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


# Window Function


def W_int(z_1, z, i, cosmo_pars=dict()):
    return interpolate_n_i['I_%s' % (str(i))](z_1)*(
        1-tilde_r(z, cosmo_pars)/tilde_r(z_1, cosmo_pars))


def Window_F(z, i, cosmo_pars=dict()):
    z_int = np.linspace(z, limits[1], 200)
    return np.trapz(W_int(z_int, z, i, cosmo_pars), z_int)


# Window function for an specific bin for redshift
# start = time.time()
# fig, ax = plt.subplots(1, 1, sharey='row', sharex='col', figsize=(10, 8))
# for z in z_arr:
#     ax.scatter(z, Window_F(z, 1), s=2.0, label='$Window Function(z)$',
#                color='mediumpurple')
# ax.set_xlabel('Redshift $z$')
# ax.set_ylabel('Window Function $\tilde{W}_{1}(z)$')
# ax.set_title('Window function for an specific bin $\tilde{W}(z)$ as a function of redshift $z$')
# end = time.time()

# print("El tiempo que se demoró es "+str(end-start)+" segundos")
# fig.show()


# fig, ax = plt.subplots(figsize=(10,8))

# for i in range(10):
#     ax.plot(z_arr, interpolate_n_i['I_%s'%(str(i))](z_arr), c='mediumpurple')
# ax.plot(z_arr, 25*n(z_arr), c='red')
# ax.set_xlabel('Redshift $z$')
# ax.set_ylabel('Number density')
# fig.show()


# Interpolator CAMB


nz = 100  # number of steps to use for the radial/redshift integration
kmax = 50   # kmax to use with k_hunit = Mpc/h

# For Limber result, want integration over \chi, from 0 to chi_*.
# so get background results to find chistar, set up a range in chi,
# and calculate corresponding redshifts
results = camb.get_background(pars)
chistar = results.conformal_time(0) - results.tau_maxvis
chis = np.linspace(0, chistar, nz)
zs = results.redshift_at_comoving_radial_distance(chis)
# Calculate array of delta_chi, and drop first and
# last points where things go singular
dchis = (chis[2:]-chis[:-2])/2
chis = chis[1:-1]
zs = zs[1:-1]

# Get the matter power spectrum interpolation object.
# Here for lensing we want the power spectrum of the Weyl potential.
PK = camb.get_matter_power_interpolator(pars,
                                        nonlinear=True,
                                        hubble_units=False,
                                        k_hunit=True,
                                        kmax=kmax,
                                        var1=model.Transfer_tot,
                                        var2=model.Transfer_tot,
                                        zmax=zs[-1])

# Have a look at interpolated power spectrum results for a range of redshifts
# Expect linear potentials to decay a bit when Lambda becomes important,
# and change from non-linear growth
# plt.figure(figsize=(10,8))
# k = np.exp(np.log(10)*np.linspace(-4, 50, 200))
# z_MPS = [0.11363636, 1.12373737, 2.38636364]
# for z in z_MPS:
#     plt.loglog(k, PK.P(z, k), color='mediumpurple')
# plt.xlim([1e-4, kmax])
# plt.xlabel('Wave-number k (h/Mpc)')
# plt.ylabel('$P_k, Mpc^3$')
# plt.legend(['z=%s'%z for z in z_bin_equi[0]])
# plt.rc('font', size=15)
# plt.rc('axes', titlesize=15)
# plt.show()


# fig, ax = plt.subplots(1, 1, sharey='row', sharex='col', figsize=(10, 8))
# k = np.exp(np.log(10)*np.linspace(-4, 50, 200))
# z_MPS = [0.11363636, 1.12373737, 2.38636364]
# for z in z_MPS:
#     ax.scatter(k, PK.P(z, k), color='mediumpurple', s=0.5)
# ax.set_xlim([1e-4, kmax])
# ax.set_xlabel('Wave-number k (h/Mpc)')
# ax.set_ylabel('$P_k, Mpc^3$')
# ax.set_xscale('log')
# ax.set_yscale('log')
# #ax.legend(z = z_bin_equi[0][0])
# fig.show()

# fig, ax = plt.subplots()

# k = np.exp(np.log(10)*np.linspace(-4, 50, 200))
# zs_mps =[0.5, 1, 1.5, 2]
# for z in zs_mps:
#     #ax.plot(kh, pk[i,:], 'x', ms=4)
#     ax.plot(k, PK.P(z, k), c='mediumpurple', label="z={:.1f}".format(z))
# ax.set_yscale('log')
# ax.set_xscale('log')
# ax.set_xbound(1e-4, 50)
# ax.set_ybound(1.5e-1, 1e5)
# ax.set_xlabel('Wave-number $k$ ($h$/Mpc)')
# ax.set_ylabel('$P_{\delta\delta}(k,z)$')
# #ax.set_title('Matter power spectrum for fixed redshifts')
# ax.legend(prop={'size': 10})
# fig.show()



# Calculation of Cosmic shear power spectrum:


# Weight function


def Weight_F(z, i, cosmo_pars=dict()):
    H0, Om, ODE, OL, Ok, wa, w0 = cosmological_parameters(cosmo_pars)
    c = const.c.value / 1000
    cte = (3/2)*((H0/c)**2)*Om
    return cte*(1 + z)*r(z, cosmo_pars)*Window_F(z, i, cosmo_pars)


def int_2(z, i, j, l, cosmo_pars=dict()):
    I1 = (Weight_F(z, i, cosmo_pars)*Weight_F(z, j, cosmo_pars))/(
          E(z, cosmo_pars)*(r(z, cosmo_pars)**2))
    k = (l + (1/2))/r(z, cosmo_pars)
    PMS = PK.P(z, k)
    return I1*PMS


def C(l, i, j, cosmo_pars=dict()):
    H0, Om, ODE, OL, Ok, wa, w0 = cosmological_parameters(cosmo_pars)
    c = const.c.value / 1000
    cte = (c/H0)
    I1 = integrate.quad(int_2, limits[0],
                        limits[1], args=(i, j, l, cosmo_pars))[0]
    return cte*I1

# FOR INDIVIDUAL I J

# start_1 = time.time()
# fig, ax = plt.subplots(1, 1, sharey='row', sharex='col', figsize=(10, 8))
# l_toplot = [138, 194, 271, 378, 529, 739, 1031, 1440, 2012]
# #l_toplot = np.arange(100, 300)
# i, j = 1, 1
# for l in l_toplot:
#     ax.scatter(l, l*(l+1)*C(l, i, j)/(2*np.pi), label="$l(l+1)/2\pi C_{1,1}$")
#     ax.scatter(l, l*(l+1)*C(l, 9, 9)/(2*np.pi), label="$l(l+1)/2\pi C_{9,9}$")
# ax.set_xlabel('Multipole l')
# ax.set_ylabel(r'$l(l+1)/2\pi C_{%s%s}^{\gamma\gamma(l)}$'%(str(i), str(j)))
# # ax.legend(['z=%s'%z for z in zplot])
# end_1 = time.time()

# print("El tiempo que se demoró es "+str(end_1-start_1)+" segundos")
# fig.legend()
# fig.show()


# cosmic_shear_array = np.zeros((200, 10, 10))
# for l in range(100, 301):
#     for i in range(10):
#         for j in range(10):
#             cosmic_shear_array[l-100, i, j] = C(l, i, j)
#             print("i: %f, j: %f" %(i, j), end = '\r')

# reshape_cosmic = np.reshape(cosmic_shear_array, (cosmic_shear_array.shape[0], -1))

# np.savetxt('quad_convergence/quad_file', reshape_cosmic)

#C_l_n = np.loadtxt('Convergence/cosmic_shear_correctls.txt')
C_l_n = np.loadtxt('Convergence/convergence_ells.txt')

C_l_i_j = np.reshape(C_l_n, (C_l_n.shape[0],
                             C_l_n.shape[1]
                             // 10, 10))


lst_C_l = dict()
for l in range(100):
    lst_C_l['C_bin_%s'%(str(l))] = C_l_n[l]


# FIHSER MATRIX
l_lst = np.linspace(10, 1500, 100)
ls = np.logspace(1, np.log10(1500), 101)
l_bins = [(ls[i], ls[i + 1]) for i in range(100)]
ls_eval = np.array([(l_bins[i][1] + l_bins[i][0]) / 2 for i in range(100)])

def shot_noise(sigma_e=0.3, ng=30):
    ng_new = ng * (180 * 60 / np.pi) ** 2
    return sigma_e ** 2 / ng_new


def Delta_l(i):
    lamba_min = np.log(l_lst[0])
    lamba_max = np.log(l_lst[-1])  # pessimist case
    N_l = 100
    delta_lambda = (lamba_max - lamba_min)/N_l
    lambda_k = lamba_min + (i - 1)*delta_lambda
    lambda_k_1 = lamba_min + i*delta_lambda
    return 10**(lambda_k_1) - 10**(lambda_k)


def Cov(i, j, m, n):
    f_sky = 0.36361
    M = np.zeros((100, 100))
    for x, l in enumerate(ls_eval):
        dl = l_bins[x][1] - l_bins[x][0]
        term_1 = (C_l_i_j[x, i, m] + shot_noise()) * (C_l_i_j[x, j, n] + shot_noise())
        term_2 = (C_l_i_j[x, i, n] + shot_noise()) * (C_l_i_j[x, j, m] + shot_noise())
        term_3 = (2*l + 1)*f_sky*dl
        M[x, x] = (term_1 + term_2) / term_3
    return M



def Obs_E(i, j):
    f_sky = 1/15000
    # delta_l = l_lst[-1] - l_lst[0]
    M = np.zeros((100, 100))
    for x, l in enumerate(np.arange(100, 200)):
        M[x, x] = np.sqrt(2/((2*l + 1)*Delta_l(i)*f_sky))*C_l_i_j[l - 100, i, j]
    return M 


def error_convergence(l_dx, i, j, dl=1490):
    dl = l_bins[l_dx][1] - l_bins[l_dx][0]
    f_sky = 0.36361
    term1 = np.sqrt(2 / ((2 * ls_eval[l_dx]  + 1) * dl * f_sky))
    term2 = C_l_i_j[i][j] + shot_noise()
    return term1 * term2 


def K_yy(z, i, j, cosmo_pars=dict()):
    H0, Om, ODE, OL, Ok, wa, w0 = cosmological_parameters(cosmo_pars)
    c = const.c.value / 1000
    cte = (H0/c)**3
    term_1 = ((3/2)*Om*(1 + z))**2
    term_2 = ((Window_F(z, i, cosmo_pars)*Window_F(z, j, cosmo_pars))
              / E(z, cosmo_pars))
    return term_1*cte*term_2


# def shot_noice(l, i, j):
#     ng = 354543085.80106884
#     sigma_e = 0.30
#     n = ng/10
#     return (sigma_e/n)*np.kron(i, j)






i, j = 0, 0
m, n = 0, 0
fig, ax = plt.subplots()

im = ax.imshow(Cov(i, j, m, n))
ax.set_title('Cov$[C_{%i%i}^{\gamma\gamma}(\ell), C_{%i%i}^{\gamma\gamma}(\ell)]$'%(i,j,m,n))
ax.set_ylim(ax.get_ylim()[::-1])
fig.colorbar(im)
#fig.show()

# fig, ax = plt.subplots()

# ell_idx = 43

# fig, ax = plt.subplots()

# im = ax.imshow(error_convergence(ell_idx))
# ax.set_title('$\Delta C_{ij}^{\gamma\gamma}(\ell=%f)$'%ls_eval[ell_idx])
# ax.set_ylim(ax.get_ylim()[::-1])
# fig.colorbar(im)

fig, ax = plt.subplots()
ltoplt = np.arange(100, 300)
# ls = np.arange(100, 1501)


for idx, ell in enumerate(ls_eval):
    lo = ax.scatter(l, l*(l+1)*C_l_i_j[idx, 1, 1]/(2*np.pi), c='mediumpurple', s=2)
    ll = ax.scatter(l, l*(l+1)*C_l_i_j[idx, 9, 9]/(2*np.pi), c='darkturquoise', s=2)
ax.set_xlim((100, 300))
ax.set_xlabel('Multipole $\ell$')
ax.set_ylabel("$C_{1,1}$ v.s $C_{9,9}$");
ax.legend((lo, ll),
           ('C_{1,1}', 'C_{9,9}'),
           scatterpoints=1,
           loc='center right',
           ncol=2,
           fontsize=10)
# fig.show()

# fig, ax = plt.subplots()

# i, j = 9, 9

# for idx, l in enumerate(ls_eval):
#     ax.scatter(l, C_l_i_j[idx, i, j], c='mediumpurple', s=0.5)
# ax.set_xlabel('Multipole $\ell$')
# ax.set_ylabel(r'$C_{%s%s}^{\gamma\gamma}(\ell)$'%(str(i),str(j)))
# ax.set_xscale('log')
# ax.set_yscale('log')
# ax.legend(prop={'size': 10})
# fig.show()

# # DERIVATES


def d_ln_E(z, dz=str(), cosmo_pars=dict()):
    H0, Om, ODE, OL, Ok, wa, w0 = cosmological_parameters(cosmo_pars)
    T1 = (1 + z)**3
    T2 = (1 + z)**2
    E_2 = E(z)**2
    e_1 = 1 + w0 + wa
    e_2 = -(3*wa*z)/(1 + z)
    E_ln = np.log(1 + z) - (z/(1+z))
    if dz == "Om":
        return (1/2)*((T1 - T2)/E_2)
    elif dz == "ODE":
        return (1/2)*(((T1**e_1)*np.exp(e_2) - T2)/E_2)
    elif dz == "w0":
        return (3/2)*((Omega_Lambda(Om)*(T1**e_1)*np.exp(e_2)*np.log(1 + z))/E_2)
    elif dz == "wa":
        return (3/2)*((Omega_Lambda(Om)*(T1**e_1)*np.exp(e_2)*E_ln)/E_2)
    else:
        return "Error"


def int_d_ln_tilder(z, dz=str(), cosmo_pars=dict()):
    H0, Om, ODE, OL, Ok, wa, w0 = cosmological_parameters(cosmo_pars)
    T1 = (1 + z)**3
    T2 = (1 + z)**2
    E_3 = E(z)**3
    e_1 = 1 + w0 + wa
    e_2 = -(3*wa*z)/(1 + z)
    E_ln = np.log(1 + z) - (z/(1+z))
    if dz == "Om":
        return (1/2)*((T1 - T2)/E_3)
    elif dz == "ODE":
        return (1/2)*(((T1**e_1)*np.exp(e_2) - T2)/E_3)
    elif dz == "w0":
        return (3/2)*((ODE*(T1**e_1)*np.exp(e_2)*np.log(1 + z))/E_3)
    elif dz == "wa":
        return (3/2)*((ODE*(T1**e_1)*np.exp(e_2)*E_ln)/E_3)
    else:
        return "Error"


def d_ln_tilder(z, dz=str(), cosmo_pars=dict()):
    H0, Om, ODE, OL, Ok, wa, w0 = cosmological_parameters(cosmo_pars)

    z_int = np.linspace(0, z, 200)
    if dz == "Om":
        return 1/2*tilde_r(z, cosmo_pars)*np.trapz(int_d_ln_tilder(z_int, "Om",cosmo_pars), z_int)
    elif dz == "ODE":
        return 1/2*tilde_r(z, cosmo_pars)*np.trapz(int_d_ln_tilder(z_int, "ODE",cosmo_pars), z_int)
    elif dz == "w0":
        return 3/2*tilde_r(z, cosmo_pars)*np.trapz(int_d_ln_tilder(z_int, "w0",cosmo_pars), z_int)
    elif dz == "wa":
        return 3/2*tilde_r(z, cosmo_pars)*np.trapz(int_d_ln_tilder(z_int, "wa",cosmo_pars), z_int)
    elif dz == "h":
        return -1/(H0/100)
    else:
        return "Error"



def int2_d_ln_W(zi, z, i, dz=str(), cosmo_pars=dict()):
    term_1 = (interpolate_n_i['I_%s' % (str(i))](zi) *
              tilde_r(z, cosmo_pars)/tilde_r(zi, cosmo_pars))
    if dz == "Om":
        term_2 = (d_ln_tilder(zi, "Om", cosmo_pars)
                  - d_ln_tilder(z, "Om", cosmo_pars))
        return term_2*term_1
    elif dz == "ODE":
        term_2 = (d_ln_tilder(zi, "ODE", cosmo_pars)
                  - d_ln_tilder(z, "ODE", cosmo_pars))
        return term_2*term_1
    elif dz == "w0":
        term_2 = (d_ln_tilder(zi, "w0", cosmo_pars)
                  - d_ln_tilder(z, "w0", cosmo_pars))
        return term_2*term_1
    elif dz == "wa":
        term_2 = (d_ln_tilder(zi, "wa", cosmo_pars)
                  - d_ln_tilder(z, "wa", cosmo_pars))
        return term_2*term_1
    elif dz == "h":
        term_2 = (d_ln_tilder(zi, "h", cosmo_pars)
                  - d_ln_tilder(z, "h", cosmo_pars))
        return term_2*term_1


def d_ln_W(z, i, dz=str(), cosmo_pars=dict()):
    I1 = Window_F(z, i, cosmo_pars)
    if dz == "Om":
        I2 = integrate.quad(
            int2_d_ln_W, z, limits[1], args=(z, i,"Om", cosmo_pars))[0]
        return I2/I1
    elif dz == "ODE":
        I2 = integrate.quad(
            int2_d_ln_W, z, limits[1], args=(z, i, "ODE", cosmo_pars))[0]
        return I2/I1
    elif dz == "w0":
        I2 = integrate.quad(
            int2_d_ln_W, z, limits[1], args=(z, i, "w0", cosmo_pars))[0]
        return I2/I1
    elif dz == "wa":
        I2 = integrate.quad(
            int2_d_ln_W, z, limits[1], args=(z, i, "wa", cosmo_pars))[0]
        return I2/I1
    elif dz == "h":
        I2 = integrate.quad(
            int2_d_ln_W, z, limits[1], args=(z, i, "h", cosmo_pars))[0]
        return I2/I1


def d_ln_K(z, i, j, dz=str(), cosmo_pars=dict()):
    H0, Om, ODE, OL, Ok, wa, w0 = cosmological_parameters(cosmo_pars)
    if dz == "Om":
        return ((2/Om) - d_ln_E(z, "Om", cosmo_pars)
                + d_ln_W(z, i, "Om", cosmo_pars)
                + d_ln_W(z, j, "Om", cosmo_pars))
    elif dz == "ODE":
        return (- d_ln_E(z, "ODE", cosmo_pars)
                + d_ln_W(z, i, "ODE", cosmo_pars)
                + d_ln_W(z, j, "ODE", cosmo_pars))
    elif dz == "w0":
        return (- d_ln_E(z, "w0", cosmo_pars)
                + d_ln_W(z, i, "w0", cosmo_pars)
                + d_ln_W(z, j, "w0", cosmo_pars))
    elif dz == "wa":
        return (- d_ln_E(z, "wa", cosmo_pars)
                + d_ln_W(z, i, "wa", cosmo_pars)
                + d_ln_W(z, j, "wa", cosmo_pars))
    elif dz == "h":
        return 3/(H0/100)


def d_K(z, i, j, dz=str(), cosmo_pars=dict()):
    term_1 = K_yy(z, i, j, cosmo_pars)
    if dz == "Om":
        return term_1*d_ln_K(z, i, j, "Om", cosmo_pars)
    elif dz == "ODE":
        return term_1*d_ln_K(z, i, j, "ODE", cosmo_pars)
    elif dz == "w0":
        return term_1*d_ln_K(z, i, j, "w0", cosmo_pars)
    elif dz == "wa":
        return term_1*d_ln_K(z, i, j, "wa", cosmo_pars)
    elif dz == "h":
        return term_1*d_ln_K(z, i, j, "h", cosmo_pars)


def d_kl(z, l, dz=str(), cosmo_pars=dict()):
    if dz == "Om":
        return (-d_ln_tilder(z, "Om", cosmo_pars)*(l + 1/2))/r(z, cosmo_pars)
    elif dz == "ODE":
        return (-d_ln_tilder(z, "ODE", cosmo_pars)*(l + 1/2))/r(z, cosmo_pars)
    elif dz == "w0":
        return (-d_ln_tilder(z, "w0", cosmo_pars)*(l + 1/2))/r(z, cosmo_pars)
    elif dz == "wa":
        return (-d_ln_tilder(z, "wa", cosmo_pars)*(l + 1/2))/r(z, cosmo_pars)
    elif dz == "h":
        return (-d_ln_tilder(z, "h", cosmo_pars)*(l + 1/2))/r(z, cosmo_pars)


def d_k_MPS(z, l):
    dk = 0.01
    k = (l + 1/2) / r(z)
    return (PK.P(z, k + dk) - PK.P(z, k)) / dk

# parametros con error asociado para obtener pendeinte de MPS


dict_MPS = dict()

dict_MPS['Omegam'] = 0.32
dict_MPS['Omegab'] = 0.05
dict_MPS['Omegade'] = 0.68
dict_MPS['Omegach2'] = 0.12055785610846023
dict_MPS['w0'] = -1.0
dict_MPS['wa'] = 0
dict_MPS['hubble'] = 0.67
dict_MPS['ns'] = 0.96
dict_MPS['sigma8'] = 0.815584
dict_MPS['gamma'] = 0.55


def d_params_PMS(z, l, dz=str(), cosmo_pars=dict()):
    dx = 0.01
    k = (l + 1/2) / r(z)
    nz = 100
    kmax = 7 
    pars_l = camb.CAMBparams()
    pars_u = camb.CAMBparams()
    if dz == "Om":
        Omegab_l, Omegab_u = (1 - dx) * dict_MPS['Omegab'], (1 + dx) * dict_MPS['Omegab']
        Omegach2_l, Omegach2_u = (1 - dx) * dict_MPS['Omegach2'], (1 + dx) * dict_MPS['Omegach2']

        pars_l.set_dark_energy(w=-1.0, wa=0, dark_energy_model='fluid')
        pars_u.set_dark_energy(w=-1.0, wa=0, dark_energy_model='fluid')
        pars_l.set_cosmology(H0=dict_MPS['hubble']*100, ombh2=Omegab_l *
                             dict_MPS['hubble']**2, omch2=Omegach2_l, tau = 0.058)
        pars_u.set_cosmology(H0=dict_MPS['hubble']*100, ombh2=Omegab_u *
                             dict_MPS['hubble']**2, omch2=Omegach2_u, tau = 0.058)
        pars.set_dark_energy(w=-1.0, wa=0, dark_energy_model='fluid')   
        pars.set_cosmology(H0=67.4, ombh2=0.02233, omch2=0.1198, omk=0, tau = 0.054)
        results_l = camb.get_results(pars_l)
        results_u = camb.get_results(pars_u)
        chistar_l = results_l.conformal_time(0) - results_l.tau_maxvis
        chistar_u = results_u.conformal_time(0) - results_u.tau_maxvis
        chis_l = np.linspace(0, chistar_l, nz)
        chis_u = np.linspace(0, chistar_u, nz)
        zs_l = results_l.redshift_at_comoving_radial_distance(chis_l)
        zs_u = results_u.redshift_at_comoving_radial_distance(chis_u)
        dchis_l = (chis_l[2:]-chis_l[:-2])/2
        dchis_u = (chis_u[2:]-chis_u[:-2])/2
        chis_l = chis_l[1:-1]
        chis_u = chis_u[1:-1]
        zs_l = zs_l[1:-1]
        zs_u = zs_u[1:-1]
        PK_l = camb.get_matter_power_interpolator(pars_l,
                                                  nonlinear=True,
                                                  hubble_units=False,
                                                  k_hunit=True,
                                                  kmax=kmax,
                                                  var1=model.Transfer_tot,
                                                  var2=model.Transfer_tot,
                                                  zmax=zs[-1])
        PK_u = camb.get_matter_power_interpolator(pars_u,
                                                  nonlinear=True,
                                                  hubble_units=False,
                                                  k_hunit=True,
                                                  kmax=kmax,
                                                  var1=model.Transfer_tot,
                                                  var2=model.Transfer_tot,
                                                  zmax=zs[-1])
    
        return (PK_u.P(z, k) - PK_l.P(z, k)) / 2*dx
    elif dz == "ODE":
        return 0

    elif dz == "w0":
        w0_l, w0_u = (1 - dx) * dict_MPS['w0'], (1 + dx) * dict_MPS['w0']
        pars_l.set_dark_energy(w=w0_l, wa=dict_MPS['wa'], dark_energy_model='fluid')
        pars_u.set_dark_energy(w=w0_u, wa=dict_MPS['wa'], dark_energy_model='fluid')
        pars_l.set_cosmology(H0=67.4, ombh2=0.02233, omch2=0.1198, omk=0, tau=0.054)
        pars_u.set_cosmology(H0=67.4, ombh2=0.02233, omch2=0.1198, omk=0, tau=0.054)
        results_l = camb.get_results(pars_l)
        results_u = camb.get_results(pars_u)
        chistar_l = results_l.conformal_time(0) - results_l.tau_maxvis
        chistar_u = results_u.conformal_time(0) - results_u.tau_maxvis
        chis_l = np.linspace(0, chistar_l, nz)
        chis_u = np.linspace(0, chistar_u, nz)
        zs_l = results_l.redshift_at_comoving_radial_distance(chis_l)
        zs_u = results_u.redshift_at_comoving_radial_distance(chis_u)
        dchis_l = (chis_l[2:]-chis_l[:-2])/2
        dchis_u = (chis_u[2:]-chis_u[:-2])/2
        chis_l = chis_l[1:-1]
        chis_u = chis_u[1:-1]
        zs_l = zs_l[1:-1]
        zs_u = zs_u[1:-1]
        PK_l = camb.get_matter_power_interpolator(pars_l,
                                                  nonlinear=True,
                                                  hubble_units=False,
                                                  k_hunit=True,
                                                  kmax=kmax,
                                                  var1=model.Transfer_tot,
                                                  var2=model.Transfer_tot,
                                                  zmax=zs[-1])
        PK_u = camb.get_matter_power_interpolator(pars_u,
                                                  nonlinear= True,
                                                  hubble_units=False,
                                                  k_hunit=True,
                                                  kmax=kmax,
                                                  var1=model.Transfer_tot,
                                                  var2=model.Transfer_tot,
                                                  zmax=zs[-1])
        return (PK_u.P(z, k) - PK_l.P(z, k)) / 2*dx

    elif dz == "wa":
        wa_l, wa_u = (1 - dx) * dict_MPS['wa'], (1 + dx) * dict_MPS['wa']
        pars_l.set_dark_energy(w=dict_MPS['w0'], wa=wa_l, dark_energy_model='fluid')
        pars_u.set_dark_energy(w=dict_MPS['w0'], wa=wa_u, dark_energy_model='fluid')
        pars_l.set_cosmology(H0=67.4, ombh2=0.02233, omch2=0.1198, omk=0, tau=0.054)
        pars_u.set_cosmology(H0=67.4, ombh2=0.02233, omch2=0.1198, omk=0, tau=0.054)
        results_l = camb.get_results(pars_l)
        results_u = camb.get_results(pars_u)
        chistar_l = results_l.conformal_time(0) - results_l.tau_maxvis
        chistar_u = results_u.conformal_time(0) - results_u.tau_maxvis
        chis_l = np.linspace(0, chistar_l, nz)
        chis_u = np.linspace(0, chistar_u, nz)
        zs_l = results_l.redshift_at_comoving_radial_distance(chis_l)
        zs_u = results_u.redshift_at_comoving_radial_distance(chis_u)
        dchis_l = (chis_l[2:]-chis_l[:-2])/2
        dchis_u = (chis_u[2:]-chis_u[:-2])/2
        chis_l = chis_l[1:-1]
        chis_u = chis_u[1:-1]
        zs_l = zs_l[1:-1]
        zs_u = zs_u[1:-1]
        PK_l = camb.get_matter_power_interpolator(pars_l,
                                                  nonlinear=True,
                                                  hubble_units=False,
                                                  k_hunit=True,
                                                  kmax=kmax,
                                                  var1=model.Transfer_tot,
                                                  var2=model.Transfer_tot,
                                                  zmax=zs[-1])
        PK_u = camb.get_matter_power_interpolator(pars_u,
                                                  nonlinear=True,
                                                  hubble_units=False,
                                                  k_hunit=True,
                                                  kmax=kmax,
                                                  var1=model.Transfer_tot,
                                                  var2=model.Transfer_tot,
                                                  zmax=zs[-1])
        return (PK_u.P(z, k) - PK_l.P(z, k)) / 2*dx

    elif dz == "h":
        hubble_l, hubble_u = (1 - dx) * dict_MPS['hubble'], (1 + dx) * dict_MPS['hubble']

        pars_l.set_dark_energy(w=-1.0, wa=0, dark_energy_model='fluid')
        pars_u.set_dark_energy(w=-1.0, wa=0, dark_energy_model='fluid')
        pars_l.set_cosmology(H0=hubble_l*100, ombh2=dict_MPS['Omegab'] *
                             hubble_l**2, omch2=dict_MPS['Omegach2']*hubble_l**2, tau=0.058)
        pars_u.set_cosmology(H0=hubble_l*100, ombh2=dict_MPS['Omegab'] *
                             hubble_u**2, omch2=dict_MPS['Omegach2']*hubble_u**2, tau=0.058)
        results_l = camb.get_results(pars_l)
        results_u = camb.get_results(pars_u)
        chistar_l = results_l.conformal_time(0) - results_l.tau_maxvis
        chistar_u = results_u.conformal_time(0) - results_u.tau_maxvis
        chis_l = np.linspace(0, chistar_l, nz)
        chis_u = np.linspace(0, chistar_u, nz)
        zs_l = results_l.redshift_at_comoving_radial_distance(chis_l)
        zs_u = results_u.redshift_at_comoving_radial_distance(chis_u)
        dchis_l = (chis_l[2:]-chis_l[:-2])/2
        dchis_u = (chis_u[2:]-chis_u[:-2])/2
        chis_l = chis_l[1:-1]
        chis_u = chis_u[1:-1]
        zs_l = zs_l[1:-1]
        zs_u = zs_u[1:-1]
        PK_l = camb.get_matter_power_interpolator(pars_l,
                                                  nonlinear=True,
                                                  hubble_units=False,
                                                  k_hunit=True,
                                                  kmax=kmax,
                                                  var1=model.Transfer_tot,
                                                  var2=model.Transfer_tot,
                                                  zmax=zs[-1])
        PK_u = camb.get_matter_power_interpolator(pars_u,
                                                  nonlinear=True,
                                                  hubble_units=False,
                                                  k_hunit=True,
                                                  kmax=kmax,
                                                  var1=model.Transfer_tot,
                                                  var2=model.Transfer_tot,
                                                  zmax=zs[-1])
        return (PK_u.P(z, k) - PK_l.P(z, k)) / 2*dx


def d_MPS(z, l, dz=str(), cosmo_pars=dict()):
    if dz == "Om":
        return d_params_PMS(z, l, "Om") + d_k_MPS(z, l)*d_kl(z, l, "Om", cosmo_pars)
    elif dz == "ODE":
        return d_params_PMS(z, l, "ODE") + d_k_MPS(z, l)*d_kl(z, l, "ODE", cosmo_pars)
    elif dz == "w0":
        return d_params_PMS(z, l, "w0") + d_k_MPS(z, l)*d_kl(z, l, "w0", cosmo_pars)
    elif dz == "wa":
        return d_params_PMS(z, l, "wa") + d_k_MPS(z, l)*d_kl(z, l, "wa", cosmo_pars)
    elif dz == "h":
        return d_params_PMS(z, l, "h") + d_k_MPS(z, l)*d_kl(z, l, "h", cosmo_pars)


def int_1_C(z, l, i, j, dz=str(), cosmo_pars=dict()):
    k = (l + 1/2) / r(z)
    term_1 = PK.P(z, k)
    if dz == "Om":
        return d_K(z, i, j, "Om", cosmo_pars)*term_1
    elif dz == "ODE":
        return d_K(z, i, j, "ODE", cosmo_pars)*term_1
    elif dz == "w0":
        return d_K(z, i, j, "w0", cosmo_pars)*term_1
    elif dz == "wa":
        return d_K(z, i, j, "wa", cosmo_pars)*term_1
    elif dz == "h":
        return d_K(z, i, j, "h", cosmo_pars)*term_1


def int_2_C(z, l, i, j, dz=str(), cosmo_pars=dict()):
    term_1 = K_yy(z, i, j)
    if dz == "Om":
        return term_1*d_MPS(z, l, "Om")
    elif dz == "ODE":
        return term_1*d_MPS(z, l, "ODE")
    elif dz == "w0":
        return term_1*d_MPS(z, l, "w0")
    elif dz == "wa":
        return term_1*d_MPS(z, l, "wa")
    elif dz == "h":
        return term_1*d_MPS(z, l, "h")


def int_3_C(z, l, i, j, dz=str(), cosmo_pars=dict()):

    k = (l + 1/2) / r(z)
    term_1 = PK.P(z, k)
    term_2 = K_yy(z, i, j)
    if dz == "Om":
        I1 = d_K(z, i, j, "Om", cosmo_pars)*term_1
        I2 = term_2*d_MPS(z, l, "Om")
        return I1 + I2
    elif dz == "ODE":
        I1 = d_K(z, i, j, "ODE", cosmo_pars)*term_1
        I2 = term_2*d_MPS(z, l, "ODE")
        return I1 + I2
    elif dz == "w0":
        I1 = d_K(z, i, j, "w0", cosmo_pars)*term_1
        I2 = term_2*d_MPS(z, l, "w0")
        return I1 + I2
    elif dz == "wa":
        I1 = d_K(z, i, j, "wa", cosmo_pars)*term_1
        I2 = term_2*d_MPS(z, l, "wa")
        return I1 + I2
    elif dz == "h":
        I1 = d_K(z, i, j, "h", cosmo_pars)*term_1
        I2 = term_2*d_MPS(z, l, "h")
        return I1 + I2


def d_C(l, i, j, dz=str(), cosmo_pars=dict()):
    if dz == "Om":
        return integrate.quad(int_3_C, limits[0], limits[1], args=(l, i, j, "Om"))[0]
    elif dz == "ODE":
        return integrate.quad(int_3_C, limits[0], limits[1], args=(l, i, j, "ODE"))[0]
    elif dz == "w0":
        return integrate.quad(int_3_C, limits[0], limits[1], args=(l, i, j, "w0"))[0]
    elif dz == "wa":
        return integrate.quad(int_3_C, limits[0], limits[1], args=(l, i, j, "wa"))[0]
    elif dz == "h":
        return integrate.quad(int_3_C, limits[0], limits[1], args=(l, i, j, "h"))[0]


# d_C_wa = np.zeros((len(ls_eval), 10, 10))

# for idx, ell in enumerate(ls_eval):
#     for i in range(10):
#         for j in range(10):
#             d_C_wa[idx, i, j] = d_C(ell, i, j, "wa")
#             print("i: %f, j: %f" %(i, j), end = '\r')

# reshape_cosmic = np.reshape(d_C_wa, (d_C_wa.shape[0], -1))
# np.savetxt('D_C/cosmic_shear_wa', reshape_cosmic)

# FISHER MATRIX

load_hubble = np.loadtxt('convergence/cosmic_shear_hubble.txt')
cosmic_shear_hubble = np.reshape(load_hubble, (load_hubble.shape[0], load_hubble.shape[1] // 10, 10))
load_ns = np.loadtxt('convergence/cosmic_shear_ns.txt')
cosmic_shear_ns = np.reshape(load_ns, (load_ns.shape[0], load_ns.shape[1] // 10, 10))
load_Omegam = np.loadtxt('convergence/cosmic_shear_omegam.txt')
cosmic_shear_Omegam = np.reshape(load_Omegam, (load_Omegam.shape[0], load_Omegam.shape[1] // 10, 10))
load_s8 = np.loadtxt('convergence/cosmic_shear_sigma8.txt')
cosmic_shear_sigma8 = np.reshape(load_s8, (load_s8.shape[0], load_s8.shape[1] // 10, 10))

dict_d_C = {'Omegam': cosmic_shear_Omegam, 'hubble': cosmic_shear_hubble, 'ns': cosmic_shear_ns, 'sigma8': cosmic_shear_sigma8}
                        #'w0': cosmic_shear_w0, 'wa': cosmic_shear_wa}


def f_c(a, b, dict=dict()):
    sum = 0
    for i in range(10):
        for j in range(10):
            for m in range(10):
                for n in range(10):
                    d_C_a = dict[a][:, i, j]
                    d_C_b = dict[b][:, m, n]
                    cov_inv = np.linalg.inv(Cov(i, j, m, n))
                    term_1 = np.dot(cov_inv, d_C_b)
                    sum += np.dot(d_C_a, term_1)
    return sum


def f_e_sum1(a, b, dict=dict()):
    sum = 0
    for i in range(10):
        for j in range(10):
            for m in range(10):
                for n in range(10):
                    d_C_a = dict[a][:,i,j]
                    d_C_b = dict[b][:,m,n]
                    OE_ni = np.linalg.inv(error_convergence(n, i))
                    OE_jm = np.linalg.inv(error_convergence(j, m))
                    term_1 = np.dot(OE_ni, d_C_b)
                    term_2 = np.dot(OE_jm, d_C_a)
                    sum += term_1 @ term_2
    return sum


F_a_b = np.array([[f_c("Omegam", "Omegam", dict_d_C), f_c("Omegam", "hubble", dict_d_C),f_c("Omegam", "ns", dict_d_C), f_c("Omegam", "sigma8", dict_d_C)],
                  [f_c("hubble", "Omegam", dict_d_C),f_c("hubble", "hubble", dict_d_C),f_c("hubble", "ns", dict_d_C),f_c("hubble", "sigma8", dict_d_C)],
                  [f_c("ns", "Omegam", dict_d_C),f_c("ns", "hubble", dict_d_C),f_c("ns", "ns", dict_d_C),f_c("ns", "sigma8", dict_d_C)],
                  [f_c("sigma8", "Omegam", dict_d_C),f_c("sigma8", "hubble", dict_d_C),f_c("sigma8", "ns", dict_d_C),f_c("sigma8", "sigma8", dict_d_C)]])
C_a_b = np.linalg.inv(F_a_b)
sigma_Om, sigma_s8, sigma_ns, sigma_h = np.sqrt(np.diag(C_a_b))

marg_sigma = np.sqrt(np.abs(np.linalg.inv(F_a_b).diagonal()))
unmarg_sigma = np.sqrt(1/F_a_b.diagonal())


fig, ax = plt.subplots()
ref_param = [0.018, 0.21, 0.035, 0.0087]
for idx in range(4):
    marg_diff = (1 - marg_sigma[idx]/ref_param[idx]) * 100
    unmarg_diff = (1 - unmarg_sigma[idx]/ref_param[idx]) * 100
    ax.scatter(idx, marg_diff, c='blue', label='Marginalised')
    ax.scatter(idx, unmarg_diff, c='red')
ax.set_label(['Marginalised', 'Unmarginalised'])
ax.set_xticks(range(4))
ax.set_xticklabels(['$\Omega_{m,0}$', '$h$', '$n_s$', '$\sigma_8$'])
ax.set_ylabel('%'+' differences on $\sigma_i$')
ax.legend(['Marginalised', 'Unmarginalised'], prop={'size':10})
fig.show()



# EXTRA Compact Notation with Kernel functions


# def K_Iy(z, i, j, cosmo_pars=dict()):
#     H0, Om, ODE, OL, Ok, wa, w0 = cosmological_parameters(cosmo_pars)
#     c = const.c.value / 1000
#     cte = (H0/c)**3
#     term_1 = ((3/2)*Om*(1*z))
#     term_2 = ((n_i(z, i)*Weight_F(z, j, cosmo_pars)) + (n_i(z, j)*Weight_F(z, i, cosmo_pars)))/tilde_r(z, cosmo_pars)
#     return term_1*cte*term_2


# def K_II(z, i, j, cosmo_pars=dict()):
#     H0, Om, ODE, OL, Ok, wa, w0 = cosmological_parameters(cosmo_pars)
#     c = const.c.value / 1000
#     cte = (H0/c)**3
#     term_1 = (n_i(z, i)*n_i(z, j)*E(z, cosmo_pars))/(tilde_r(z, cosmo_pars)**2)
#     return term_1*cte


# def P_DI(z, k, cosmo_pars=dict()):
#     H0, Om, ODE, OL, Ok, wa, w0 = cosmological_parameters(cosmo_pars)
#     A_ia = 1.72
#     C_ia = 0.0134
#     cte = -A_ia*C_ia*Om
#     term_1 = 1/D(z, cosmo_pars)
#     return cte*term_1*PK.P(z, k)


# def P_II(z, k, cosmo_pars=dict()):
#     H0, Om, ODE, OL, Ok, wa, w0 = cosmological_parameters(cosmo_pars)
#     A_ia = 1.72
#     C_ia = 0.0134
#     cte = -A_ia*C_ia*Om
#     term_1 = 1/D(z, cosmo_pars)
#     return ((cte*term_1)**2)*PK.P(z, k)


# def int_3(z, i, j, l, cosmo_pars=dict()):
#     k = (l + (1/2))/r(z, cosmo_pars)
#     I1 = K_yy(z, i, j, cosmo_pars)*PK.P(z, k)
#     I2 = K_Iy(z, i, j, cosmo_pars)*P_DI(z, k)
#     I3 = K_II(z, i, j, cosmo_pars)*P_II(z, k)
#     return I1 + I2 + I3