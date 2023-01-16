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


def Omega_Lambda(Omega_m, Omega_b, Omega_v):
    """La funcion Omega_Lambda nos entregara este parametro en base a los que
    tenemos, para esto tambien debemos calcular Omega c, un parametro que
    no se utilizara, por lo que no es necesario almacenar"""
    Omega_c = Omega_m - Omega_b
    OL = 1 - Omega_c - Omega_b - Omega_v
    return OL


def Omega_K_0(Omega_DE, Omega_m):
    """Omega_K_0 nos entrega este parametro que es depende de Omega DE y
    Omega m, en el caso de del modelo ΛCDM este valor es cero"""
    OK = 1 - (Omega_DE + Omega_m)
    return OK


def cosmological_parameters(cosmo_pars=dict()):
    """cosmological_parameters extrae los parametros necesarios para las
    el calculo de funciones E(z) y D(z), concadena estos parametros de manera
    que sean facil de utilizar, el default son los parametros de Planck 2018"""
    H0 = cosmo_pars.get('H0', params_P18['H0'])
    Om = cosmo_pars.get('Om', params_P18['Om'])
    Ob = cosmo_pars.get('Ob', params_P18['Ob'])
    Ov = cosmo_pars.get('Ov', params_P18['Ov'])
    ODE = cosmo_pars.get('ODE', params_P18['ODE'])
    OL = Omega_Lambda(Om, Ob, Ov)
    OK = Omega_K_0(ODE, Om)
    wa = cosmo_pars.get('wa', params_P18['w_a'])
    w0 = cosmo_pars.get('w0', params_P18['w_0'])
    return H0, Om, ODE, OL, OK, wa, w0


def E_arb(z, cosmo_pars=dict()):
    """E_arb es la función E(z) arbitraria para cualquier modelo cosmologico"""
    H0, Om, ODE, OL, Ok, wa, w0 = cosmological_parameters(cosmo_pars)
    exp = np.exp(-3*wa*(z/1+z))
    ind = 1 + wa + w0
    E = np.sqrt(Om*(1+z)**3 + ODE*((1+z)**(3*ind))*exp + Ok*(1+z)**2)
    return E


def E(z, cosmo_pars=dict()):
    """E es la función E(z) para el caso w0 = -1 y wa = 0"""
    H0, Om, ODE, OL, Ok, wa, w0 = cosmological_parameters(cosmo_pars)
    E = np.sqrt(Om*(1+z)**3 + OL + Ok*(1+z)**2)
    return E


# comoving distance to an object redshift z

def f_integral(z, cosmo_pars=dict()):
    """f_integral define la funcion dentro de la integral
    ocupada para el calculo de r(z)"""
    return 1/E_arb(z, cosmo_pars)


def r(z, cosmo_pars=dict()):
    """r calcula comoving distnace to an objecto redshift"""
    c = const.c.value
    cte = c/params_P18['H0']  # h^-1 Mpc
    int = integrate.quad(f_integral, 0, z, args=cosmo_pars)
    r = cte*int[0]
    return r


# transverse comoving distance


def D(z, cosmo_pars=dict()):
    """La funcion D calcula transverse comoving distance para los distintos
    casos de el parametro Omgea_K_0"""
    c = const.c.value
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
fig, ax = plt.subplots(1, 1, sharey='row', sharex='col', figsize=(10, 8))

# Proper distance dependent on redshift plot
ax.plot(z_arr, E_arb(z_arr), label='$E(z)$', color='mediumpurple')
ax.set_xlabel('Redshift $z$')
ax.set_ylabel('$E(z)$')
ax.set_title('Proper distance $E(z) as a function of redshift $z$')
# plt.show()

# Comoving distance to an object redshift z plot

fig, ax = plt.subplots(1, 1, sharey='row', sharex='col', figsize=(10, 8))
for z in z_arr:
    ax.scatter(z, r(z), s=1.0, label='$r(z)$', color='mediumpurple')
ax.set_xlabel('Redshift $z$')
ax.set_ylabel('$Comoving distance r(z)$')
ax.set_title('Comoving distance $r(z)$ as a function of redshift $z$')
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
    c = const.c.value
    H0, Om, ODE, OL, Ok, wa, w0 = cosmological_parameters(cosmo_pars)
    cte = c/H0
    return r(z, cosmo_pars)/cte

# Photometric redshift estimates:


def n(z):
    zm = 0.9  # median redshift, value given by Euclid Red Book
    z0 = zm/(np.sqrt(2))
    frac = z/z0
    return (frac**2)*np.exp(-frac**(3/2))


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

# FUNCIONES MOMENTANEAS DEBO CORREGIR Y REVISARLAS


def n_i(z, i):
    I1 = integrate.quad(int_1, z_bin.bin_edges[i],
                        z_bin.bin_edges[i+1], args=z)[0]
    I2 = integrate.nquad(int_1, [[z_bin.bin_edges[i], z_bin.bin_edges[i+1]],
                         [limits[0], limits[1]]])[0]

    return I1/I2

# Window Function


def W_int(z_1, z, i):
    return n_i(z_1, i)*(1-(tilde_r(z)/tilde_r(z_1)))


def Window_F(z, i):
    return integrate.quad(W_int, z, limits[1], args=(z, i))[0]


# Window function for an specific bin for redshift
# start = time.time()
# fig, ax = plt.subplots(1, 1, sharey='row', sharex='col', figsize=(10, 8))
# for z in z_arr:
#     ax.scatter(z, Window_F(z, 10), s=2.0, label='$Window Function(z)$',
#                color='mediumpurple')
# ax.set_xlabel('Redshift $z$')
# ax.set_ylabel('Window Function $\tilde{W}_{10}(z)$')
# ax.set_title('Window function for an specific bin $\tilde{W}(z)$ as a function of redshift $z$')
# end = time.time()

# print("El tiempo que se demoró es "+str(end-start)+" segundos")
# plt.show()


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
df = pd.DataFrame(list_of_PMS,
                  columns=['kh', 'z', 'pk', 'nonlinear_kh',
                           'nonlinear_z', 'nonlinear_pk'])

# Print data.
df.to_csv('PMS_params.txt', sep='\t')

for i in range(len(z_bin_equi[0])):
    list = []
    for z in z_bin[0]:
        list.append([n_i(z, i), z])
    df_1 = pd.DataFrame(list, columns=['bin number density', 'z'])

df_1.to_csv('Bin_number_d.txt', sep='\t')



# Plotting matter power spectrum

# plt.figure(figsize=(8, 5))
# for i, (redshift, line) in enumerate(zip(z_bin_equi[0], ['-', '--'])):
#     plt.loglog(kh_nonlin, pk_nonlin[i, :], color='r', ls=line)
# plt.xlabel('k/h Mpc')
# plt.legend(['linear','non-linear'], loc='lower left')
# plt.title('Matter power at z=%s and z= %s'%tuple(z_bin_equi[0]))
# plt.show()

# For calculating large-scale structure and lensing results yourself,
# get a power spectrum interpolation object.
# In this example we calculate the CMB lensing potential power
# spectrum using the Limber approximation,
# using PK=camb.get_matter_power_interpolator() function.
# calling PK(z, k) will then get power spectrum at any k and redshift z in range.

nz = 100  # number of steps to use for the radial/redshift integration
kmax = 2   # kmax to use

# For Limber result, want integration over \chi, from 0 to chi_*.
# so get background results to find chistar, set up a range in chi,
# and calculate corresponding redshifts
results = camb.get_background(pars)
chistar = results.conformal_time(0) - results.tau_maxvis
chis = np.linspace(0, chistar, nz)
zs = results.redshift_at_comoving_radial_distance(chis)
# Calculate array of delta_chi, and drop first and last points where things go singular
dchis = (chis[2:]-chis[:-2])/2
chis = chis[1:-1]
zs = zs[1:-1]

# Get the matter power spectrum interpolation object.
# Here for lensing we want the power spectrum of the Weyl potential.
PK = camb.get_matter_power_interpolator(pars,
                                        nonlinear=True,
                                        hubble_units=False,
                                        k_hunit=False,
                                        kmax=kmax,
                                        var1=model.Transfer_Weyl,
                                        var2=model.Transfer_Weyl,
                                        zmax=zs[-1])

# Have a look at interpolated power spectrum results for a range of redshifts
# Expect linear potentials to decay a bit when Lambda becomes important,
# and change from non-linear growth

# plt.figure(figsize=(8,5))

# l_toplot = [138, 194, 271, 378, 529, 739, 1031, 1440, 2012]

# for l in l_toplot:
#     k = (l + (1/2))/r(z)
#     for z in z_bin_equi[0]:
#         plt.loglog(k, PK.P(z, k), color='mediumpurple')
# plt.xlim([1e-4,kmax])
# plt.xlabel('k Mpc')
# plt.ylabel('$P_\Psi\, Mpc^{-3}$')
# plt.legend(['z=%s'%z for z in z_bin_equi[0]])
# plt.show()

# Calculation of Cosmic shear power spectrum:

# Weight function


def Weight_F(z, i, cosmo_pars=dict()):
    H0, Om, ODE, OL, Ok, wa, w0 = cosmological_parameters(cosmo_pars)
    c = const.c.value
    cte = (3/2)*((H0/c)**2)*Om
    return cte*(1 + z)*r(z, cosmo_pars)*Window_F(z, i)


def int_2(z, i, j, l, cosmo_pars=dict()):
    I1 = (Weight_F(z, i, cosmo_pars)*Weight_F(z, j, cosmo_pars))/(E(z, cosmo_pars)*(r(z, cosmo_pars)**2))
    k = (l + (1/2))/r(z, cosmo_pars)
    PMS = PK.P(z, k)
    return I1*PMS


def C(l, i, j, cosmo_pars=dict()):
    H0, Om, ODE, OL, Ok, wa, w0 = cosmological_parameters(cosmo_pars)
    c = const.c.value
    cte = (c/H0)
    I1 = integrate.quad(int_2, limits[0], limits[1], args=(i, j, l, cosmo_pars))[0]
    return cte*I1

# FOR INDIVIDUAL I J

# start_1 = time.time()
# fig, ax = plt.subplots(1, 1, sharey='row', sharex='col', figsize=(10, 8))
# l_toplot = [138, 194, 271, 378, 529, 739, 1031, 1440, 2012] # segun yo se plotea z_arr o la lista de los bins
# i, j = 2, 9 
# for l in l_toplot:
#     ax.scatter(l, C(l, i, j))
# ax.set_xlabel('Multipole l')
# ax.set_ylabel(r'$C_{%s%s}^{\gamma\gamma(l)}$'%(str(i), str(j)))
# # ax.legend(['z=%s'%z for z in zplot])
# end_1 = time.time()

# print("El tiempo que se demoró es "+str(end_1-start_1)+" segundos")
# plt.show()




# Compact Notation with Kernel functions

def K_yy(z, i, j, cosmo_pars=dict()):
    H0, Om, ODE, OL, Ok, wa, w0 = cosmological_parameters(cosmo_pars)
    c = const.c.value
    cte = (H0/c)**3
    term_1 = ((3/2)*Om*(1*z))**2
    term_2 = (Weight_F(z, i, cosmo_pars)*Weight_F(z, j, cosmo_pars))/E(z, cosmo_pars)
    return term_1*cte*term_2


def K_Iy(z, i, j, cosmo_pars=dict()):
    H0, Om, ODE, OL, Ok, wa, w0 = cosmological_parameters(cosmo_pars)
    c = const.c.value
    cte = (H0/c)**3
    term_1 = ((3/2)*Om*(1*z))
    term_2 = ((n_i(z, i)*Weight_F(z, j, cosmo_pars)) + (n_i(z, j)*Weight_F(z, i, cosmo_pars)))/tilde_r(z, cosmo_pars)
    return term_1*cte*term_2


def K_II(z, i, j, cosmo_pars=dict()):
    H0, Om, ODE, OL, Ok, wa, w0 = cosmological_parameters(cosmo_pars)
    c = const.c.value
    cte = (H0/c)**3
    term_1 = (n_i(z, i)*n_i(z, j)*E(z, cosmo_pars))/(tilde_r(z, cosmo_pars)**2)
    return term_1*cte


def P_DI(z, k, cosmo_pars=dict()):
    H0, Om, ODE, OL, Ok, wa, w0 = cosmological_parameters(cosmo_pars)
    A_ia = 1.72
    C_ia = 0.0134
    cte = -A_ia*C_ia*Om
    term_1 = 1/D(z, cosmo_pars)
    return cte*term_1*PK.P(z, k)


def P_II(z, k, cosmo_pars=dict()):
    H0, Om, ODE, OL, Ok, wa, w0 = cosmological_parameters(cosmo_pars)
    A_ia = 1.72
    C_ia = 0.0134
    cte = -A_ia*C_ia*Om
    term_1 = 1/D(z, cosmo_pars)
    return ((cte*term_1)**2)*PK.P(z, k)


def int_3(z, i, j, l, cosmo_pars=dict()):
    k = (l + (1/2))/r(z, cosmo_pars)
    I1 = K_yy(z, i, j, cosmo_pars)*PK.P(z, k)
    I2 = K_Iy(z, i, j, cosmo_pars)*P_DI(z, k)
    I3 = K_II(z, i, j, cosmo_pars)*P_II(z, k)
    return I1 + I2 + I3



plt.figure(figsize=(8,5))
z_equi = [0.0010, 0.42, 0.56, 0.68, 0.79, 0.90, 1.02, 1.15, 1.32, 1.58, 2.50]
l_toplot = np.linspace(100,2000,200)

for z in z_equi:
    k = (l + (1/2))/r(z)
    plt.loglog(k, PK.P(z, k), color='mediumpurple')
plt.xlim([1e-4,kmax])
plt.xlabel('k Mpc')
plt.ylabel('$P_\Psi\, Mpc^{-3}$')
plt.legend(['z=%s'%z for z in z_bin_equi[0]])
plt.show()