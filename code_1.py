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
    H0 = cosmo_pars.get('H0', params_CAMB['H0'])
    Om = cosmo_pars.get('Om', params_CAMB['Om'])
    Ob = cosmo_pars.get('Ob', params_CAMB['Ob'])
    Ov = cosmo_pars.get('Ov', params_CAMB['Ov'])
    ODE = cosmo_pars.get('ODE', params_CAMB['ODE'])
    OL = Omega_Lambda(Om, Ob, Ov)
    OK = Omega_K_0(ODE, Om)
    wa = cosmo_pars.get('wa', params_CAMB['w_a'])
    w0 = cosmo_pars.get('w0', params_CAMB['w_0'])
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
    c = const.c.value / 1000
    cte = c/params_P18['H0']  # h^-1 Mpc
    int = integrate.quad(f_integral, 0, z, args=cosmo_pars)
    r = cte*int[0]
    return r


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
z_equi = [0.0010, 0.42, 0.56, 0.68, 0.79, 0.90, 1.02, 1.15, 1.32, 1.58, 2.50] # This are the values from the paper

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

# FUNCIONES MOMENTANEAS DEBO CORREGIR Y REVISARLAS

z_equipop_bins = [(0.001, 0.42), (0.42, 0.56), (0.56, 0.68), (0.68, 0.79), (0.79, 0.90),
                    (0.90, 1.02), (1.02, 1.15) ,(1.15, 1.32), (1.32, 1.58), (1.58, 2.50)]


def n_i(z, i):
    ith_bin = z_equipop_bins[i]
    zi_l, zi_u = ith_bin
    I1 = integrate.quad(int_1, zi_l,
                        zi_u, args=z)[0]
    I2 = integrate.nquad(int_1, [[zi_l, zi_u],
                         [limits[0], limits[1]]])[0]

    return I1/I2




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

# Storage number density

z_list = z_bin[1]

lst_0 = []
for z in z_list:
    lst_0.append([z, n_i(z, 0)])
n_0 = pd.DataFrame(lst_0)
n_0.to_csv('Bin_number_d_0.txt', sep='\t')

lst_1 = []
for z in z_list:
    lst_1.append([z, n_i(z, 1)])
n_1 = pd.DataFrame(lst_1)
n_1.to_csv('Bin_number_d_1.txt', sep='\t')

lst_2 = []
for z in z_list:
    lst_2.append([z, n_i(z, 2)])
n_2 = pd.DataFrame(lst_2)
n_2.to_csv('Bin_number_d_2.txt', sep='\t')

lst_3 = []
for z in z_list:
    lst_3.append([z, n_i(z, 3)])
n_3 = pd.DataFrame(lst_3)
n_3.to_csv('Bin_number_d_3.txt', sep='\t')

lst_4 = []
for z in z_list:
    lst_4.append([z, n_i(z, 4)])
n_4 = pd.DataFrame(lst_4)
n_4.to_csv('Bin_number_d_4.txt', sep='\t')

lst_5 = []
for z in z_list:
    lst_5.append([z, n_i(z, 5)])
n_5 = pd.DataFrame(lst_5)
n_5.to_csv('Bin_number_d_5.txt', sep='\t')

lst_6 = []
for z in z_list:
    lst_6.append([z, n_i(z, 6)])
n_6 = pd.DataFrame(lst_6)
n_6.to_csv('Bin_number_d_6.txt', sep='\t')

lst_7 = []
for z in z_list:
    lst_7.append([z, n_i(z, 7)])
n_7 = pd.DataFrame(lst_7)
n_7.to_csv('Bin_number_d_7.txt', sep='\t')

lst_8 = []
for z in z_list:
    lst_8.append([z, n_i(z, 8)])
n_8 = pd.DataFrame(lst_8)
n_8.to_csv('Bin_number_d_8.txt', sep='\t')

lst_9 = []
for z in z_list:
    lst_9.append([z, n_i(z, 9)])
n_9 = pd.DataFrame(lst_9)
n_9.to_csv('Bin_number_d_9.txt', sep='\t')


lst_n_i = dict()

lst_n_i["bin_0"] = np.array(lst_0)
lst_n_i["bin_1"] = np.array(lst_1)
lst_n_i["bin_2"] = np.array(lst_2)
lst_n_i["bin_3"] = np.array(lst_3)
lst_n_i["bin_4"] = np.array(lst_4)
lst_n_i["bin_5"] = np.array(lst_5)
lst_n_i["bin_6"] = np.array(lst_6)
lst_n_i["bin_7"] = np.array(lst_7)
lst_n_i["bin_8"] = np.array(lst_8)
lst_n_i["bin_9"] = np.array(lst_9)



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

def W_int(z_1, z, i):
    return interpolate_n_i['I_%s'%(str(i))](z_1)*(1-(tilde_r(z)/tilde_r(z_1)))


def Window_F(z, i):
    return integrate.quad(W_int, z, limits[1], args=(z, i))[0]


fig, ax = plt.subplots()

for i in range(10):
    ax.plot(z_arr, interpolate_n_i['I_%s'%(str(i))](z_arr), c='b')
ax.plot(z_arr, 25*n(z_arr), c='red')
ax.set_xlabel('Redshift $z$')
ax.set_ylabel('Number density')
fig.show()


# Interpolator CAMB

# For calculating large-scale structure and lensing results yourself,
# get a power spectrum interpolation object.
# In this example we calculate the CMB lensing potential power
# spectrum using the Limber approximation,
# using PK=camb.get_matter_power_interpolator() function.
# calling PK(z, k) will then get power spectrum at any k and redshift z in range.

nz = 100  # number of steps to use for the radial/redshift integration
kmax = 7   # kmax to use with k_hunit = Mpc/h

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
                                        k_hunit=True,
                                        kmax=kmax,
                                        var1=model.Transfer_tot,
                                        var2=model.Transfer_tot,
                                        zmax=zs[-1])

# Have a look at interpolated power spectrum results for a range of redshifts
# Expect linear potentials to decay a bit when Lambda becomes important,
# and change from non-linear growth



# Calculation of Cosmic shear power spectrum:


# Weight function


def Weight_F(z, i, cosmo_pars=dict()):
    H0, Om, ODE, OL, Ok, wa, w0 = cosmological_parameters(cosmo_pars)
    c = const.c.value / 1000
    cte = (3/2)*((H0/c)**2)*Om
    return cte*(1 + z)*r(z, cosmo_pars)*Window_F(z, i)


def int_2(z, i, j, l, cosmo_pars=dict()):
    I1 = (Weight_F(z, i, cosmo_pars)*Weight_F(z, j, cosmo_pars))/(E(z, cosmo_pars)*(r(z, cosmo_pars)**2))
    k = (l + (1/2))/r(z, cosmo_pars)
    PMS = PK.P(z, k)
    return I1*PMS


def C(l, i, j, cosmo_pars=dict()):
    H0, Om, ODE, OL, Ok, wa, w0 = cosmological_parameters(cosmo_pars)
    c = const.c.value / 1000
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
    c = const.c.value / 1000
    cte = (H0/c)**3
    term_1 = ((3/2)*Om*(1*z))**2
    term_2 = (Weight_F(z, i, cosmo_pars)*Weight_F(z, j, cosmo_pars))/E(z, cosmo_pars)
    return term_1*cte*term_2


def K_Iy(z, i, j, cosmo_pars=dict()):
    H0, Om, ODE, OL, Ok, wa, w0 = cosmological_parameters(cosmo_pars)
    c = const.c.value / 1000
    cte = (H0/c)**3
    term_1 = ((3/2)*Om*(1*z))
    term_2 = ((n_i(z, i)*Weight_F(z, j, cosmo_pars)) + (n_i(z, j)*Weight_F(z, i, cosmo_pars)))/tilde_r(z, cosmo_pars)
    return term_1*cte*term_2


def K_II(z, i, j, cosmo_pars=dict()):
    H0, Om, ODE, OL, Ok, wa, w0 = cosmological_parameters(cosmo_pars)
    c = const.c.value / 1000
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




# plt.figure(figsize=(8,5))
# k = np.exp(np.log(10)*np.linspace(-4, 7, 200))
# for z in z_bin_equi[0]:
#     plt.loglog(k, PK.P(z, k), color='mediumpurple')
# plt.xlim([1e-4,kmax])
# plt.xlabel('Wave-number k (h/Mpc)')
# plt.ylabel('$P_k, Mpc^3$')
# plt.legend(['z=%s'%z for z in z_bin_equi[0]])
# plt.show()
