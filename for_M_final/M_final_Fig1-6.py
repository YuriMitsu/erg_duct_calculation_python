# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
fce = 5.0 * 2 #kHz
f = 0.02 * fce
fpe = 10. * fce
theta = np.arange(0.,89.5,0.01)

n = fpe / np.sqrt( fce * f * ( np.cos( theta / 180 * np.pi ) - f / fce ) )
nx = n * np.cos( theta / 180 * np.pi )
ny = n * np.sin( theta / 180 * np.pi )
linewidth = 2

# %%
fig = plt.figure(figsize=(3,2))
ax = fig.add_subplot(111)
ax.plot(ny, nx, linewidth=linewidth, c='k')
ax.plot(-ny, nx, linewidth=linewidth, c='k')
ax.annotate('', xy=[0, 0], xytext=[0, 40],
            arrowprops=dict(facecolor='black', lw=linewidth, arrowstyle='<|-')
           )
ax.annotate('', xy=[-150, 0.7], xytext=[150, 0.7],
            arrowprops=dict(facecolor='black', lw=linewidth, arrowstyle='-')
           )
# ax.grid(which='both')
ax.set_xlim(-150,150)
ax.set_ylim(0, 50)
ax.set_xticklabels([''])
ax.set_yticklabels([''])
# ax.set_yticks([200, 225, 250, 275])
# ax.set_xticks([0, 40, 80])
# ax.set_ylim(100,175)
# ax.set_yticks([100, 125, 150, 175])

spines = 2
ax.spines["top"].set_linewidth(spines)
ax.spines["left"].set_linewidth(spines)
ax.spines["bottom"].set_linewidth(spines)
ax.spines["right"].set_linewidth(spines)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.axis('off')

# ax.tick_params(width=1.5, labelsize=14)
# ax.grid()
fig.tight_layout()
fig.savefig('/Users/ampuku/Desktop/Smith+Fig3_1', dpi=600)



# %%
fce = 5.0 * 2 #kHz
f = 0.52 * fce
fpe = 10. * fce
theta = np.arange(0.,58.,0.01)

n = fpe / np.sqrt( fce * f * ( np.cos( theta / 180 * np.pi ) - f / fce ) )
nx = n * np.cos( theta / 180 * np.pi )
ny = n * np.sin( theta / 180 * np.pi )
linewidth = 2

# %%
fig = plt.figure(figsize=(3,1.5))
ax = fig.add_subplot(111)
ax.plot(ny, nx, linewidth=linewidth, c='k')
ax.plot(-ny, nx, linewidth=linewidth, c='k')
ax.annotate('', xy=[0, 0], xytext=[0, 90],
            arrowprops=dict(facecolor='black', lw=linewidth, arrowstyle='<|-')
           )
ax.annotate('', xy=[-150, 0.7], xytext=[150, 0.7],
            arrowprops=dict(facecolor='black', lw=linewidth, arrowstyle='-')
           )
# ax.grid(which='both')
ax.set_xlim(-150,150)
ax.set_ylim(0, 80)
ax.set_xticklabels([''])
ax.set_yticklabels([''])
# ax.set_yticks([200, 225, 250, 275])
# ax.set_xticks([0, 40, 80])
# ax.set_ylim(100,175)
# ax.set_yticks([100, 125, 150, 175])

spines = 2
ax.spines["top"].set_linewidth(spines)
ax.spines["left"].set_linewidth(spines)
ax.spines["bottom"].set_linewidth(spines)
ax.spines["right"].set_linewidth(spines)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.axis('off')

# ax.tick_params(width=1.5, labelsize=14)
# ax.grid()
fig.tight_layout()
fig.savefig('/Users/ampuku/Desktop/Smith+Fig3_2', dpi=600)



# %%
