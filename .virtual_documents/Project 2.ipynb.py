import jax.numpy as jnp
import project_2 as p2
import seaborn as sns
from tqdm import tqdm
import imp
imp.reload(p2)
sns.set(context='paper',style='ticks')


#Initial conditions
Ig = 0.75 * jnp.exp(jnp.deg2rad(180)*1j)
Eg = 1.0315 * jnp.exp(jnp.deg2rad(14.2)*1j)
Im = 0.75 #* np.exp(0*1j)
Em = 1/(p2.bm*-1j)*(-1*Im) +1
Pgm = 0.75
Pmm = 0.75
V1 = 1/(p2.b*-1j)*Im + 1
V2 = 1.0*jnp.exp(0*1j)

#Parameters that were not stated anywhere in the project.
omega0 = 2*jnp.pi*60









measurements  = jnp.array([[1.000, 0.100, 1.000, 0.000, 0.750, 0.000, -0.750, 0.000,-0.750, 0.000, 0.750, 0.000, 0.000, 0.000],
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


P = 10e6*jnp.eye(8)
W = (1/(0.01**2))*jnp.eye(measurements.shape[1])


x_k = jnp.array([jnp.real(V1),jnp.imag(V1),jnp.cos(jnp.angle(Eg)),jnp.sin(jnp.angle(Eg)),jnp.angle(Eg),0,jnp.real(V2),jnp.imag(V2),jnp.cos(jnp.angle(Em)),jnp.sin(jnp.angle(Em)),jnp.angle(Em),0])
print(x_k)





X = []
tol = 1e-6
k = 0
x_t_h = x_k


print(p2.g_cons(x_k,x_t_h))


jnp.abs(p2.Em)


g = p2.g_cons(x_k,x_t_h)
print(g)


x_k





def d_vec(x_k,x_t_h,z):
    h_x = p2.h(x_k)-z
    g_x_k = p2.g_cons(x_k,x_t_h)
    d = jnp.concatenate([h_x,g_x_k])
    return d

def update_rule(x_k,H,G,W,P,d):
    left = jnp.linalg.inv(jnp.vstack([H,G]).T @ jnp.block([[W,jnp.zeros((14,8))],[jnp.zeros((8,14)),P]])@jnp.vstack([H,G]))
    right = (jnp.vstack([H,G]).T@jnp.block([[W,jnp.zeros((14,8))],[jnp.zeros((8,14)),P]]))@d
    x_k_1 = x_k - left @ right
    #x_k_1 = x_k - jnp.linalg.inv(H.T@W@H + G.T@P@G)@(jnp.hstack([H.T@W,G.T@P]))@d
    return x_k_1



residuals = []
stds = []
for t,z in enumerate(measurements):
    print("==============================t= ",t)
    for i in tqdm(range(10)):
        H = p2.measurement_jacobian(x_k)
        G = p2.constraint_jacobian(x_k,x_t_h)
        d = d_vec(x_k,x_t_h,z)
        x_k = update_rule(x_k,H,G,W,P,d)
    print(x_k)
    #Save residuals and standard deviations
    stds.append(std(H,G,W,P))
    residuals.append(p2.h(x_k)-z)
    X.append(x_k)
    x_t_h = x_k


d


import matplotlib.pyplot as plt
X = jnp.asarray(X)
names = ['$v_{1,r}$','$v_{1,i}$','$v_{2,r}$','$v_{2,i}$','$i_{1,r}$','$i_{1,i}$','$i_{2,r}$','$i_{2,i}$','$i_{gr}$','$i_{gi}$','$i_{mr}$','$i_{mi}$','$\omega_g$','$\omega_m$']
for i in range(12):
    name = names[i]
    plt.plot(X[:,i],label=name)
plt.legend()


figscale = 2
fig,axes = plt.subplots(nrows=2,ncols=3,figsize=(figscale*3.5,figscale*3.5/1.61828),constrained_layout=True,sharex=True)
axes[1,0].set_xlabel("time t (t/60)s")
axes[1,1].set_xlabel("time t (t/60)s")
axes[1,2].set_xlabel("time t (t/60)s")


#Generator and motor speed
axes[0,0].plot([p2.omegag(x) for x in X],label="$\omega_g$")
axes[0,0].plot([p2.omegam(x) for x in X],label="$\omega_m$")
axes[0,0].legend()
axes[0,0].set_title("Angular Velocities")
axes[0,0].set_ylabel("rad/s")
axes[0,0].grid()

#Deltas
axes[0,1].plot([p2.deltag(x) for x in X],label="$\delta_g$")
axes[0,1].plot([p2.deltam(x) for x in X],label="$\delta_m$")
axes[0,1].legend()
axes[0,1].set_title("Rotor Angles")
axes[0,1].set_ylabel("Radians")
axes[0,1].grid()

#Voltages
axes[0,2].plot([p2.v1r(x) for x in X],label="$V_{1,r}$")
axes[0,2].plot([p2.v2r(x) for x in X],label="$V_{2,r}$")
axes[0,2].plot([p2.v1i(x) for x in X],label="$V_{1,i}$")
axes[0,2].plot([p2.v2i(x) for x in X],label="$V_{2,i}$")
axes[0,2].legend()
axes[0,2].set_title("Real/Imag Part of Voltages")
axes[0,2].set_ylabel("Volts")
axes[0,2].grid()

#Voltage mags
axes[1,0].plot([jnp.abs(p2.v1r(x) + p2.v1i(x)*1j) for x in X],label=r"$|\tilde{V}_1|$")
axes[1,0].plot([jnp.abs(p2.v2r(x) + p2.v2i(x)*1j) for x in X],label=r"$|\tilde{V}_{2}|$")
axes[1,0].set_ylabel("V RMS")
axes[1,0].set_title("Line Voltage Mags.")
axes[1,0].set_ylim(0.5,1.1)
axes[1,0].legend()
axes[1,0].grid()

#Voltage angles
axes[1,1].plot([jnp.angle(p2.v1r(x) + p2.v1i(x)*1j) for x in X],label=r"$\angle\tilde{V}_1$")
axes[1,1].plot([jnp.angle(p2.v2r(x) + p2.v2i(x)*1j)for x in X],label=r"$\angle\tilde{V}_{2}$")
axes[1,1].set_ylabel("Radians")
axes[1,1].legend()
axes[1,1].grid()
axes[1,1].set_title("Line Voltage Angles")


#Voltage angles
axes[1,2].grid()
axes[1,2].plot([p2.cg(x) for x in X],label=r'$c_g$')
axes[1,2].plot([p2.cm(x) for x in X],label=r'$c_m$')
axes[1,2].plot([p2.sg(x) for x in X],'--',label=r'$s_g$')
axes[1,2].plot([p2.sm(x) for x in X],'--',label=r'$s_m$')
axes[1,2].set_title("Trig. Relationships")
axes[1,2].legend()


#save
plt.savefig("Figures/results.png",dpi=400)


import pandas as pd
import numpy as np

STDS = np.asarray(stds)

df = pd.DataFrame()
names = ['$v_{1,r}$','$v_{1,i}$','$c_g$','$s_g$','$\delta_g$','$\omega_g$','$v_{2,r}$','$v_{2,i}$','$c_{m}$','$s_m$','$\delta_m$','$\omega_m$']
stds_list = []
labels = []
for j,s_j in enumerate(STDS.T):
    labels.append([names[j] for i in range(len(s_j))])
    stds_list.append(s_j)

df['Standard Deviation'] = np.asarray(stds).flatten()
df['names'] = np.asarray(labels).flatten()


fig,ax = plt.subplots(figsize=(2*3.5,2*3.5/1.61828))
plt.grid()
sns.boxplot(data=df,x='names',y='Standard Deviation',hue='names')
plt.legend(frameon=False)
plt.savefig("Figures/std.png",dpi=400)



from scipy.stats import chi2
def goodness_of_fit(J,df):
    return 1 - chi2.cdf(J,df)





sns.violinplot(data=df,x='')


def std(H,G,W,P):
    left = jnp.linalg.inv(jnp.vstack([H,G]).T @ jnp.block([[W,jnp.zeros((14,8))],[jnp.zeros((8,14)),P]])@jnp.vstack([H,G]))
    stds = []
    for i,s_i in enumerate(left):
        stds.append(jnp.sqrt(s_i[i]))
    return stds


info_matrix = jnp.linalg.inv(H.T @ W @ H)
stds = []
for i,s_i in enumerate(info_matrix):
    stds.append(jnp.sqrt(s_i[i]))
stds



