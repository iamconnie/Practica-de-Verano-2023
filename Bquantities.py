# importancion de paquetes necesarios para el código

import sys
import os
import numpy as np

# instalacion de camb 
camb_path = os.path.realpath(os.path.join(os.getcwd(),'..'))
sys.path.insert(0,camb_path)

import camb as camb
# print('Using CAMB %s installed at %s'%(camb.__version__,os.path.dirname(camb.__file__)))


# Creación de funciones E(z) y D(z), a manera de prueba se ocuparan los
# valores de Planck 2016 para testear las funciones

params_P18 = dict()

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


def Omega_Lambda(Omega_m, Omega_b, Omega_v):
    Omega_c = Omega_m - Omega_b
    OL = 1 - Omega_c - Omega_b - Omega_v
    return OL


def Omega_K_0(Omega_DE, Omega_m):
    OK = 1 + (Omega_DE + Omega_m)
    return OK


def cosmological_parameters(cosmo_pars=dict()):
    "cosmological_parameters extrae los parametros necesarios para las"
    "el calculo de funciones E(z) y D(z), concadena estos parametros de manera"
    "que sean facil de utilizar, el default son los parametros de Planck 2018"
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
    "E_arb es la función E(z) arbitraria para cualquier modelo cosmologico"
    H0, Om, ODE, OL, Ok, wa, w0 = cosmological_parameters(cosmo_pars)
    exp = np.exp(-3*wa(z/1+z))
    ind = 1 + wa + w0
    E = np.sqrt(Om*(1+z)**3 + ODE*((1+z)**(3*ind))*exp + Ok(1+z)**2)
    return E


def E(z, cosmo_pars=dict()):
    "E es la función E(z) para el caso w0 = -1 y wa = 0"
    H0, Om, ODE, OL, Ok, wa, w0 = cosmological_parameters(cosmo_pars)
    E = np.sqrt(Om*(1+z)**3 + OL + Ok(1+z)**2)
    return E
