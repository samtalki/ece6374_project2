import numpy as np
from jax import jacfwd
import jax.numpy as jnp
import project_2 as p2
import seaborn as sns
from tqdm import tqdm
import imp
imp.reload(p2)
sns.set(context='paper',style='darkgrid')


bg = 4.9 
Hg = 2.5 #seconds
Hm = 1.8
Sm = 1.0
bm = 5.8
Sg = 1
b = 7.5
g = 0.95
omega0 = 2*np.pi*60
#By fiat
Pgm = 0.75
Pmm = 0.75
Im = 0.75 * np.exp(0*1j)
Ig = 0.75 * np.exp(np.deg2rad(180)*1j)
Eg = 1.0315 * np.exp(np.deg2rad(14.2)*1j)
Em = 1/(bm*-1j)*(-1*Im) +1
V1 = 1/(b*-1j)*Im + 1
V2 = 1.0*np.exp(0*1j)



cycles = [i for i in range(14)]
times = [n*(1/60) for n in cycles]


measurements  = np.array([[1.000, 0.100, 1.000, 0.000, 0.750, 0.000, -0.750, 0.000,-0.750, 0.000, 0.750, 0.000, 0.000, 0.000],
                          [1.000, 0.100, 1.000, 0.000, 0.750, 0.000, -0.750, 0.000, -0.750, 0.000, 0.750, 0.000, 0.000, 0.000,],
                          [0.993, 0.030, 0.987, -0.114, 1.079, -0.041, -1.079, 0.041, -1.079, 0.041, 0.141, 0.068, -0.392, -1.092],
                          [0.991, 0.013, 0.983, -0.135, 1.111, -0.060, -1.111, 0.060, -1.111, 0.060, 0.177, 0.068, -0.811, -2.146],
                          [0.988, -0.014, 0.975, -0.169, 1.160, -0.095, -1.160, 0.095,-1.160, 0.095, 0.233, 0.065, -1.278, -3.125],
                          [0.982, -0.052, 0.963, -0.215, 1.219, -0.148, -1.219, 0.148,-1.219, 0.148, 0.305, 0.056, -1.813, -4.001],
                          [0.974, -0.101, 0.944, -0.271, 1.281, -0.222, -1.281, 0.222, -1.281, 0.222, 0.384, 0.036, -2.425, -4.757],
                          [0.960, -0.159, 0.918, -0.337, 1.335, -0.318, -1.335, 0.318, -1.335, 0.318, 0.464, 0.002, -3.118, -5.387],
                          [0.941, -0.226, 0.883, -0.409, 1.372, -0.436, -1.372, 0.436, -1.372, 0.436, 0.534, -0.047, -3.888, -5.896],
                          [0.915, -0.302, 0.838, -0.487, 1.382, -0.572, -1.382, 0.572, -1.382, 0.572, 0.586, -0.110, -4.723, -6.300],
                          [0.916, -0.334, 0.844, -0.482, 1.106, -0.538, -1.106, 0.538, -1.106, 0.538, 1.106, -0.538, -5.609, -6.622],
                          [0.877, -0.424, 0.790, -0.565, 1.055, -0.652, -1.055, 0.652, -1.055, 0.652, 1.055, -0.652, -6.175, -5.836],
                          [0.832, -0.512, 0.734, -0.640, 0.959, -0.735, -0.959, 0.735, -0.959, 0.735, 0.959, -0.735, -6.730, -5.067],
                          [0.780, -0.597, 0.676, -0.707, 0.828, -0.778, -0.828, 0.778, -0.828, 0.778, 0.828, -0.778, -7.226, -4.377]
                         ])


P = 100e6*np.eye(8)
W = (1/(0.01**2))*np.eye(measurements.shape[1])


def d_vec(x_k,x_k_h,z):
    h_x = np.asarray(p2.h(x_k))-z
    g_x_k = p2.g_cons(x_k,x_k_h)
    d = np.concatenate([h_x,g_x_k])
    return d

def update_rule(x_k,H,G,W,P,d):
    x_k_1 = x_k - np.linalg.inv(H.T@W@H + G.T@P@G)@(np.hstack([H.T@W,G.T@P]))@d
    return x_k_1



#cg = cos(angle(eg))
#cm = cos(angle(em))
#sg = sin(angle(eg))
#sm = sin(angle(em))
x_k = np.array([np.real(V1),np.imag(V1),np.cos(np.angle(Eg)),np.sin(np.angle(Eg)),np.angle(Eg),0,np.real(V2),np.imag(V2),np.cos(np.angle(Em)),np.sin(np.angle(Em)),np.angle(Em),0])
print(x_k)


print(p2.g_cons(x_k,x_k))


np.abs(p2.Em)


X = []
tol = 1e-6
k = 0
# z = measurements[0,:]

# H = p2.measurement_jacobian(x_k)
# G = p2.constraint_jacobian(x_k,x_k)
# d = d_vec(x_k,x_k,z)
#x_k = update_rule(x_k,H,G,W,P,d)
x_k_h = x_k


g = p2.g_cons(x_k,x_k_h)
print(g)


x_k



for z,t in enumerate(measurements):
    print("==============================t= ",t)
    for i in tqdm(range(10)):
        
        #print(x_k)
        H = p2.measurement_jacobian(x_k)
        G = p2.constraint_jacobian(x_k,x_k_h)
        #print(H,G)
        d = d_vec(x_k,x_k_h,z)
        x_k = update_rule(x_k,H,G,W,P,d)
       
        #print(d[14:])
    print(x_k)
    X.append(x_k)
    x_k_h = x_k


import matplotlib.pyplot as plt
X = np.asarray(X)
names = ['$v_{1,r}$','$v_{1,i}$','$v_{2,r}$','$v_{2,i}$','$i_{1,r}$','$i_{1,i}$','$i_{2,r}$','$i_{2,i}$','$i_{gr}$','$i_{gi}$','$i_{mr}$','$i_{mi}$','$\omega_g$','$\omega_m$']
for i in range(12):
    name = names[i]
    plt.plot(X[:,i],label=name)
plt.legend()


fig,axes = plt.subplots(nrows=2,ncols=3,figsize=(2.5*3.5,2.5*3.5/1.61828),constrained_layout=True,sharex=True)
axes[1,0].set_xlabel("time t (t/60)s")
axes[1,1].set_xlabel("time t (t/60)s")

#Generator and motor speed
axes[0,0].plot([p2.omegag(x) for x in X],label="$\omega_g$")
axes[0,0].plot([p2.omegam(x) for x in X],label="$\omega_m$")
axes[0,0].legend()
axes[0,0].set_title("Angular Velocities")
#Deltas
axes[0,1].plot([p2.deltag(x) for x in X],label="$\delta_g$")
axes[0,1].plot([p2.deltam(x) for x in X],label="$\delta_m$")
axes[0,1].legend()
axes[0,1].set_title("Angles")
#Voltages
axes[0,2].plot([p2.v1r(x) for x in X],label="$V_{1,r}$")
axes[0,2].plot([p2.v2r(x) for x in X],label="$V_{2,r}$")
axes[0,2].plot([p2.v1i(x) for x in X],label="$V_{1,i}$")
axes[0,2].plot([p2.v2i(x) for x in X],label="$V_{2,i}$")
axes[0,2].legend()
axes[0,2].set_title("Voltages")

#Currents
axes[1,0].plot([p2.v1r(x) for x in X],label="$V_{1,r}$")
axes[1,0].plot([p2.v2r(x) for x in X],label="$V_{2,r}$")
axes[1,0].plot([p2.v1i(x) for x in X],label="$V_{1,i}$")
axes[1,0].plot([p2.v2i(x) for x in X],label="$V_{2,i}$")
axes[1,0].legend()
axes[1,0].set_title("Voltages")



print(X[0])






