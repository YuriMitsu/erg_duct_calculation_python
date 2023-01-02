# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
%matplotlib inline

# %%
path = '/Users/ampuku/Documents/duct/code/python/for_paper_figure/f_Ne_f_B_data43/'
Nmin, Nmax = 242.-4.2, 255.+4.4

Bdata = pd.read_csv(path+'Bv_Bobs_data.csv', header=None)
Bv = Bdata[0]
Bobs = Bdata[1]
Ndata = pd.read_csv(path+'plotf_Ne0_Ne1_data.csv', header=None)
plotf = Ndata[0]
Ne0 = Ndata[1]
Ne1 = Ndata[2]
Fdata = pd.read_csv(path+'eqfce_data.csv', header=None)
fce_eq = Fdata[0]

# deff1 = np.array(plotf[abs(Ne1-Nmin)==sorted(abs(Ne1-Nmin))[0]])
# deff2 = np.array(plotf[abs(Ne1-Nmax)==sorted(abs(Ne1-Nmax))[0]])
# vlines = [deff1[0], deff2[0]]

plt.rcParams["font.size"] = 20
fig = plt.figure(figsize=(12, 10))

# ax5 = fig.add_subplot(2,4,3)
ax5 = fig.add_subplot(2,1,1)
ax5.plot(plotf,Ne0, color='k', linewidth=2)
ax5.plot(plotf,Ne1, color='k', linewidth=2, linestyle='dashed')
ax5.axhspan(Nmin, Nmax, color="b", alpha=0.3)
ax5.set_xlim(0.0, 1.0)
# ax5.set_ylim(100, 190)
ax5.set_ylim(10, 500)
# ax5.set_xlabel("")
# ax5.set_ylabel("Ne [/cc]")
# ax5.set_xticks([1.,1.5,2.,2.5,3.,3.5,4.])
# ax5.set_xticklabels([])



# ax6 = fig.add_subplot(2,4,7)
ax6 = fig.add_subplot(2,1,2)
ax6.plot(Bv/14.299986,Bobs, color='k', linewidth=2)
ax6.set_xlim(0.0, 1.0)
ax6.set_ylim(0., 0.05)
ax6.set_xlabel("frequency [kHz]")
ax6.set_ylabel("OFA-B [$\mathrm{pT^2/Hz}$]")


# spines = 2
# for ax in [ax5, ax6]:
#     ax.spines["top"].set_linewidth(spines)
#     ax.spines["left"].set_linewidth(spines)
#     ax.spines["bottom"].set_linewidth(spines)
#     ax.spines["right"].set_linewidth(spines)
#     ax.minorticks_on()
#     ax.tick_params(which='major', width=2, length=6)
#     ax.tick_params(which='minor', width=2, length=3)
#     # ax.grid()

# for xx in vlines:
#     ax5.vlines(xx, 90, 300, colors='k', linestyle='dashed', linewidth=2)
#     ax6.vlines(xx, -0.1, 0.2, colors='k', linestyle='dashed', linewidth=2)
# ax6.vlines(fce_eq, -0.1, 0.2, colors='k', linestyle='dotted', linewidth=2)

# fig.tight_layout()

# fig.savefig('/Users/ampuku/Documents/duct/Fig/_paper_figure/f_Ne_f_B_plot3.png', dpi=300)

# %%
