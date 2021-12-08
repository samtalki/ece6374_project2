import numpy as np
import pandas as pd
from jax import jacfwd
import jax.numpy as jnp


#Physical parameters of the system
bg = 4.9 
Hg = 2.5 #seconds
Sg = 1
b = 7.5
g = 0.95
bm = 5.8
Hm = 1.8
Sm = 1.0

#Initial conditions
Ig = 0.75 * np.exp(np.deg2rad(180)*1j)
Eg = 1.0315 * np.exp(np.deg2rad(14.2)*1j)
Im = 0.75 #* np.exp(0*1j)
Em = 1/(bm*-1j)*(-1*Im) +1
Pgm = 0.75
Pmm = 0.75

#Parameters that were not stated anywhere in the project.
omega0 = 2*np.pi*60





def measurement_jacobian(x):
    H = jacfwd(h)(jnp.asarray(x))
    return H


def constraint_jacobian(x,x_t_h):
    G = jacfwd(g_cons)(jnp.asarray(x),x_t_h)
    return G

def h(x):
    h1 = v1r(x) #V1_r
    h2 = v1i(x) #V1_i
    h3 = v2r(x) #V2_r
    h4 = v2i(x) #V2_i
    h5 = b*(v1i(x)-v2i(x)) #I1_r
    h6 = -b*(v1r(x)-v2r(x)) #I1_i
    h7 = b*(v2i(x)-v1i(x)) #I2_r
    h8 = -b*(v2r(x)-v1r(x)) #I2_i
    h9 = bg*(v1i(x)-np.abs(Eg)*sg(x)) #Ig_r
    h10 = -bg*(v1r(x)-np.abs(Eg)*cg(x)) #Ig_i
    h11 = bm*(v2i(x) - np.abs(Em)*sm(x)) #Im_r
    h12 = -bm*(v2r(x) - np.abs(Em)*cm(x)) #Im_i
    h13 = omegag(x) #omega_g
    h14 = omegam(x) #omega_m
    return jnp.asarray([h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14])


def v1r(x):
    return x[0]
def v1i(x):
    return x[1]
def cg(x):
    return x[2]
def sg(x):
    return x[3]
def deltag(x):
    return x[4]
def omegag(x):
    return x[5]
def v2r(x):
    return x[6]
def v2i(x):
    return x[7]
def cm(x):
    return x[8]
def sm(x):
    return x[9]
def deltam(x):
    return x[10]
def omegam(x):
    return x[11]


def dyn_cons_1(x,x_t_h,h=1/60):
    val = 2*Hg*Sg/omega0 * (omegag(x) - omegag(x_t_h)) - h*Pgm + \
        bg*np.abs(Eg)*((h/3)*v1r(x)*sg(x) + (h/6)*v1r(x_t_h)*sg(x) + (h/6)*v1r(x)*sg(x_t_h) + (h/3)*v1r(x_t_h)*sg(x_t_h)) - \
            bg*np.abs(Eg)*((h/3)*v1i(x)*cg(x) + (h/6)*v1i(x_t_h)*cg(x) + (h/6)*v1i(x)*cg(x_t_h) + (h/3)*v1i(x_t_h)*cg(x_t_h))
    return val

def dyn_cons_2(x,x_t_h,h=1/60):
    val = deltag(x) - deltag(x_t_h) - (h/2)*omegag(x) -(h/2)*omegag(x_t_h)
    return val

def dyn_cons_3(x,x_t_h,h=1/60):
    val = cg(x) - cg(x_t_h) +(h/3)*omegag(x)*sg(x) +(h/6)*omegag(x_t_h)*sg(x) + (h/6)*omegag(x)*sg(x_t_h) + (h/3)*omegag(x_t_h)*sg(x_t_h)
    return val

def dyn_cons_4(x,x_t_h,h=1/60):
    val = sg(x) -sg(x_t_h) - (h/3)*omegag(x)*cg(x) - (h/6)*omegag(x_t_h)*cg(x) - (h/6)*omegag(x)*cg(x_t_h) - (h/3)*omegag(x_t_h)*cg(x_t_h)
    return val

def dyn_cons_5(x,x_t_h,h=1/60):
    val = 2*Hm*Sm/omega0 * (omegam(x) - omegam(x_t_h)) + h*Pmm + \
        bm*np.abs(Em)*((h/3)*v2r(x)*sm(x) + (h/6)*v2r(x_t_h)*sm(x) + (h/6)*v2r(x)*sm(x_t_h) + (h/3)*v2r(x_t_h)*sm(x_t_h)) - \
            bm*np.abs(Em)*((h/3)*v2i(x)*cm(x) + (h/6)*v2i(x_t_h)*cm(x) + (h/6)*v2i(x)*cm(x_t_h) + (h/3)*v2i(x_t_h)*cm(x_t_h))
    return val

def dyn_cons_6(x,x_t_h,h=1/60):
    val = deltam(x) - deltam(x_t_h) - (h/2)*omegam(x) -(h/2)*omegam(x_t_h)
    return val

def dyn_cons_7(x,x_t_h,h=1/60):
    val = cm(x) - cm(x_t_h) +(h/3)*omegam(x)*sm(x) +(h/6)*omegam(x_t_h)*sm(x) + (h/6)*omegam(x)*sm(x_t_h) + (h/3)*omegam(x_t_h)*sm(x_t_h)
    return val

def dyn_cons_8(x,x_t_h,h=1/60):
    val = sm(x) -sm(x_t_h) - (h/3)*omegam(x)*cm(x) - (h/6)*omegam(x_t_h)*cm(x) - (h/6)*omegam(x)*cm(x_t_h) - (h/3)*omegam(x_t_h)*cm(x_t_h)
    return val


def g_cons(x_t,x_t_h):
    g_functions  = [dyn_cons_1,dyn_cons_2,dyn_cons_3,dyn_cons_4,dyn_cons_5,dyn_cons_6,dyn_cons_7,dyn_cons_8]
    g = [g_i(x_t,x_t_h) for g_i in g_functions]
    return jnp.asarray(g) 