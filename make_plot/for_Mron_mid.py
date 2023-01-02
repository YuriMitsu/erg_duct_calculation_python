# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
fce = 8*2 #kHz
f = 10
theta = np.arange(0.,80.,1.)
# kperp = np.arange(0.0, 0.004, 0.0001)

kpara_linear = 0.0002 + 0.0001 * f

# b1 = (9.1093 * 10**(-31)) / (1.25 * 10**(-6)) / (1.6 * 10**(-19))**2
b1_theta = (9.1093 * 10**(-31)) / (1.25 * 10**(-6)) / (1.6 * 10**(-19))**2

kperp_theta = kpara_linear * np.tan( theta / 180 * np.pi ) # tan([radian])
b2 = - kperp_theta**2 + fce / f * kpara_linear * np.sqrt( kpara_linear**2 + kperp_theta**2 ) - kpara_linear**2
Ne = b1_theta * b2 / 10**6 #cm-3

# b2 = - kperp**2 + fce / f * kpara_linear * np.sqrt( kpara_linear**2 + kperp**2 ) - kpara_linear**2
# Ne_kperp = b1 * b2 / 10**(6) #cm-3

# %%
# kpara_linear = 0.0002 + 0.0001 * f
# fce = 20*2, f = 10
# で使用

fig = plt.figure(figsize=(3,2.5))
ax = fig.add_subplot(111)
ax.plot(theta, Ne, linewidth=1)
# ax.grid(which='both')
ax.set_xlim(0,max(theta))
ax.set_xticks([0, 40, 80])
ax.set_ylim(100,175)
ax.set_yticks([100, 125, 150, 175])

spines = 2
ax.spines["top"].set_linewidth(spines)
ax.spines["left"].set_linewidth(spines)
ax.spines["bottom"].set_linewidth(spines)
ax.spines["right"].set_linewidth(spines)

ax.tick_params(width=1.5, labelsize=14)

fig.tight_layout()



# %%
fig = plt.figure(figsize=(2.9,2.5))
ax = fig.add_subplot(111)
ax.plot(theta, Ne, linewidth=1)
# ax.grid(which='both')
ax.set_xlim(0,max(theta))
ax.set_xticks([0, 40, 80])
ax.set_ylim(0,30)
ax.set_yticks([0, 10, 20, 30])

spines = 2
ax.spines["top"].set_linewidth(spines)
ax.spines["left"].set_linewidth(spines)
ax.spines["bottom"].set_linewidth(spines)
ax.spines["right"].set_linewidth(spines)

ax.tick_params(width=1.5, labelsize=14)

fig.tight_layout()
# %%
