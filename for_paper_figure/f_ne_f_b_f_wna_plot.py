# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %%
fce_ave_all = [14.105383, 13.445186, 30.402452, 14.299986]  # 規格化用

# %%
plt.rcParams["font.size"] = 25
fig = plt.figure(figsize=(30, 9), facecolor='white')


# %%
path = '/Users/ampuku/Documents/duct/code/python/for_paper_figure/f_Ne_f_B_data1/'
Nmin, Nmax = 280., 296.

Bdata = pd.read_csv(path+'Bv_Bobs_data.csv', header=None)
Bv = Bdata[0]
Bobs = Bdata[1]
Ndata = pd.read_csv(path+'plotf_Ne0_Ne1_data.csv', header=None)
plotf = Ndata[0]
Ne0 = Ndata[1]
Ne1 = Ndata[2]
Fdata = pd.read_csv(path+'eqfce_data.csv', header=None)
fce_eq = Fdata[0]
fce_ave = fce_ave_all[0]

deff = abs(Ne0 - Nmax)
vlines = [plotf[deff == sorted(deff)[0]], plotf[deff == sorted(deff)[1]]]

# %%

ax1 = fig.add_subplot(4, 4, 9)
ax1.plot(plotf/fce_ave, Ne0, color='k', linewidth=2)
ax1.plot(plotf/fce_ave, Ne1, color='k', linewidth=2, linestyle='solid')
ax1.axhspan(Nmin, Nmax, color="r", alpha=0.3)
ax1.set_xlim(0.05, 0.30)
ax1.set_ylim(270, 320)
ax1.set_xlabel("")
ax1.set_ylabel("Ne [/cc]")
# ax1.set_xticks([1.,1.5,2.,2.5,3.,3.5,4.])
ax1.set_xticklabels([])


ax2 = fig.add_subplot(4, 4, 13)
ax2.plot(Bv, Bobs, color='k', linewidth=2)
ax2.set_xlim(0.05*fce_ave, 0.30*fce_ave)
ax2.set_ylim(0., 0.1)
ax2.set_xlabel("frequency [kHz]")
# ax2.set_xlabel("f / fc")
ax2.set_ylabel("OFA-B [$\mathrm{pT^2/Hz}$]")


spines = 2
for ax in [ax1, ax2]:
    ax.spines["top"].set_linewidth(spines)
    ax.spines["left"].set_linewidth(spines)
    ax.spines["bottom"].set_linewidth(spines)
    ax.spines["right"].set_linewidth(spines)
    ax.minorticks_on()
    ax.tick_params(which='major', width=2, length=6)
    ax.tick_params(which='minor', width=2, length=3)
    # ax.grid()

for xx in vlines:
    ax1.vlines(xx/fce_ave, 270, 340, colors='k', linestyle='solid', linewidth=2)
    ax2.vlines(xx, -0.1, 0.2, colors='k', linestyle='solid', linewidth=2)
ax2.vlines(fce_eq, -0.1, 0.2, colors='k', linestyle='dotted', linewidth=2)

# fig.tight_layout()

# fig.savefig('/Users/ampuku/Documents/duct/Fig/_paper_figure/f_Ne_f_B_plot1.png', dpi=300)


# %%

path = '/Users/ampuku/Documents/duct/code/python/for_paper_figure/f_Ne_f_B_data2/'
Nmin, Nmax = 218., 260.

Bdata = pd.read_csv(path+'Bv_Bobs_data.csv', header=None)
Bv = Bdata[0]
Bobs = Bdata[1]
Ndata = pd.read_csv(path+'plotf_Ne0_Ne1_data.csv', header=None)
plotf = Ndata[0]
Ne0 = Ndata[1]
Ne1 = Ndata[2]
Fdata = pd.read_csv(path+'eqfce_data.csv', header=None)
fce_eq = Fdata[0]
fce_ave = fce_ave_all[1]

deff = abs(Ne0 - Nmin)
vlines1 = [plotf[deff == sorted(deff)[0]], plotf[deff == sorted(deff)[1]]]
deff = abs(Ne1 - Nmax)
vlines2 = [plotf[deff == sorted(deff)[0]]]

# %%
# plt.rcParams["font.size"] = 20
# fig = plt.figure(figsize=(12, 10))

# ax3 = fig.add_subplot(2,4,2)
ax3 = fig.add_subplot(4, 4, 10)
ax3.plot(plotf/fce_ave, Ne0, color='k', linewidth=2)
ax3.plot((plotf/fce_ave)[plotf/fce_ave < 0.5], Ne1[plotf/fce_ave < 0.5], color='k', linewidth=2, linestyle='dashed')
ax3.axhspan(Nmin, Nmax, color="b", alpha=0.3)
ax3.set_xlim(0.25, 0.5)
ax3.set_ylim(200, 280)
ax3.set_xlabel("")
ax3.set_ylabel("Ne [/cc]")
# ax3.set_xticks([1.,1.5,2.,2.5,3.,3.5,4.])
ax3.set_xticklabels([])


# ax4 = fig.add_subplot(2,4,6)
ax4 = fig.add_subplot(4, 4, 14)
ax4.plot(Bv, Bobs, color='k', linewidth=2)
ax4.set_xlim(0.25*fce_ave, 0.5*fce_ave)
ax4.set_ylim(0., 0.08)
ax4.set_xlabel("frequency [kHz]")
# ax4.set_xlabel("f / fc")
ax4.set_ylabel("OFA-B [$\mathrm{pT^2/Hz}$]")


spines = 2
for ax in [ax3, ax4]:
    ax.spines["top"].set_linewidth(spines)
    ax.spines["left"].set_linewidth(spines)
    ax.spines["bottom"].set_linewidth(spines)
    ax.spines["right"].set_linewidth(spines)
    ax.minorticks_on()
    ax.tick_params(which='major', width=2, length=6)
    ax.tick_params(which='minor', width=2, length=3)
    # ax.grid()

for xx in vlines1:
    ax3.vlines(xx/fce_ave, 180, 340, colors='k', linestyle='solid', linewidth=2)
    ax4.vlines(xx, -0.1, 0.2, colors='k', linestyle='solid', linewidth=2)
for xx in vlines2:
    ax3.vlines(xx/fce_ave, 180, 340, colors='k', linestyle='dashed', linewidth=2)
    ax4.vlines(xx, -0.1, 0.2, colors='k', linestyle='dashed', linewidth=2)
ax4.vlines(fce_eq, -0.1, 0.2, colors='k', linestyle='dotted', linewidth=2)

# fig.tight_layout()

# fig.savefig('/Users/ampuku/Documents/duct/Fig/_paper_figure/f_Ne_f_B_plot2.png', dpi=300)


# %%

path = '/Users/ampuku/Documents/duct/code/python/for_paper_figure/f_Ne_f_B_data3/'
# Nmin, Nmax = 128., 166.
Nmin, Nmax = 128., 230.

Bdata = pd.read_csv(path+'Bv_Bobs_data.csv', header=None)
Bv = Bdata[0]
Bobs = Bdata[1]
Ndata = pd.read_csv(path+'plotf_Ne0_Ne1_data.csv', header=None)
plotf = Ndata[0]
Ne0 = Ndata[1]
Ne1 = Ndata[2]
Fdata = pd.read_csv(path+'eqfce_data.csv', header=None)
fce_eq = Fdata[0]
fce_ave = fce_ave_all[2]

deff1 = np.array(plotf[abs(Ne1-Nmin) == sorted(abs(Ne1-Nmin))[0]])
deff2 = np.array(plotf[abs(Ne1-Nmax) == sorted(abs(Ne1-Nmax))[0]])
vlines = [deff1[0], deff2[0]]

# %%
# plt.rcParams["font.size"] = 20
# fig = plt.figure(figsize=(12, 10))

# ax5 = fig.add_subplot(2,4,3)
ax5 = fig.add_subplot(4, 4, 11)
ax5.plot(plotf/fce_ave, Ne0, color='k', linewidth=2)
ax5.plot(plotf/fce_ave, Ne1, color='k', linewidth=2, linestyle='dashed')
ax5.axhspan(Nmin, Nmax, color="b", alpha=0.3)
ax5.set_xlim(0.02, 0.17)
# ax5.set_ylim(100, 190)
ax5.set_ylim(100, 250)
ax5.set_xlabel("")
ax5.set_ylabel("Ne [/cc]")
# ax5.set_xticks([1.,1.5,2.,2.5,3.,3.5,4.])
ax5.set_xticklabels([])


# ax6 = fig.add_subplot(2,4,7)
ax6 = fig.add_subplot(4, 4, 15)
ax6.plot(Bv, Bobs, color='k', linewidth=2)
ax6.set_xlim(0.02*fce_ave, 0.17*fce_ave)
ax6.set_ylim(0., 0.04)
ax6.set_xlabel("frequency [kHz]")
# ax6.set_xlabel("f / fc")
ax6.set_ylabel("OFA-B [$\mathrm{pT^2/Hz}$]")


spines = 2
for ax in [ax5, ax6]:
    ax.spines["top"].set_linewidth(spines)
    ax.spines["left"].set_linewidth(spines)
    ax.spines["bottom"].set_linewidth(spines)
    ax.spines["right"].set_linewidth(spines)
    ax.minorticks_on()
    ax.tick_params(which='major', width=2, length=6)
    ax.tick_params(which='minor', width=2, length=3)
    # ax.grid()

for xx in vlines:
    ax5.vlines(xx/fce_ave, 90, 300, colors='k', linestyle='dashed', linewidth=2)
    ax6.vlines(xx, -0.1, 0.2, colors='k', linestyle='dashed', linewidth=2)
ax6.vlines(fce_eq, -0.1, 0.2, colors='k', linestyle='dotted', linewidth=2)

# fig.tight_layout()

# fig.savefig('/Users/ampuku/Documents/duct/Fig/_paper_figure/f_Ne_f_B_plot3.png', dpi=300)

# %%


# %%

path = '/Users/ampuku/Documents/duct/code/python/for_paper_figure/f_Ne_f_B_data4/'
Nmin, Nmax = 242., 255.
Nmin_err, Nmax_err = 242.-4.2, 255.+4.4

Bdata = pd.read_csv(path+'Bv_Bobs_data.csv', header=None)
Bv = Bdata[0]
Bobs = Bdata[1]
Ndata = pd.read_csv(path+'plotf_Ne0_Ne1_data.csv', header=None)
plotf = Ndata[0]
Ne0 = Ndata[1]
Ne1 = Ndata[2]
Fdata = pd.read_csv(path+'eqfce_data.csv', header=None)
fce_eq = Fdata[0]
fce_ave = fce_ave_all[3]

deff1 = abs(Ne0 - Nmin_err)
vlines_s = [plotf[deff1 == sorted(deff1)[0]], plotf[deff1 == sorted(deff1)[1]]]
deff2 = np.array(plotf[abs(Ne1-Nmax_err) == sorted(abs(Ne1-Nmax_err))[0]])
vlines_d = [deff2[0]]


# %%
# plt.rcParams["font.size"] = 20
# fig = plt.figure(figsize=(12, 10))

ax7 = fig.add_subplot(4, 4, 12)
ax7.plot(plotf/fce_ave, Ne0, color='k', linewidth=2)
ax7.plot((plotf/fce_ave)[plotf/fce_ave < 0.5], Ne1[plotf/fce_ave < 0.5], color='k', linewidth=2, linestyle='dashed')
ax7.axhspan(Nmin_err, Nmax_err, color="b", alpha=0.15)
ax7.axhspan(Nmin, Nmax, color="b", alpha=0.15)
ax7.set_xlim(0.3, 0.6)
ax7.set_ylim(220, 280)
ax7.set_xlabel("")
ax7.set_ylabel("Ne [/cc]")
# ax7.set_xticks([1.,1.5,2.,2.5,3.,3.5,4.])
ax7.set_xticklabels([])


ax8 = fig.add_subplot(4, 4, 16)
ax8.plot(Bv, Bobs, color='k', linewidth=2)
ax8.set_xlim(0.3*fce_ave, 0.6*fce_ave)
ax8.set_ylim(0., 0.11)
ax8.set_xlabel("frequency [kHz]")
# ax8.set_xlabel("f / fc")
ax8.set_ylabel("WFC-B [$\mathrm{pT^2/Hz}$]")


spines = 2
for ax in [ax7, ax8]:
    ax.spines["top"].set_linewidth(spines)
    ax.spines["left"].set_linewidth(spines)
    ax.spines["bottom"].set_linewidth(spines)
    ax.spines["right"].set_linewidth(spines)
    ax.minorticks_on()
    ax.tick_params(which='major', width=2, length=6)
    ax.tick_params(which='minor', width=2, length=3)
    # ax.grid()

for xx in vlines_s:
    ax7.vlines(xx/fce_ave, 100, 300, colors='k', linestyle='solid', linewidth=2)
    ax8.vlines(xx, -0.1, 0.2, colors='k', linestyle='solid', linewidth=2)
for xx in vlines_d:
    ax7.vlines(xx/fce_ave, 100, 300, colors='k', linestyle='dashed', linewidth=2)
    ax8.vlines(xx, -0.1, 0.2, colors='k', linestyle='dashed', linewidth=2)
ax8.vlines(fce_eq, -0.1, 0.2, colors='k', linestyle='dotted', linewidth=2)

# fig.tight_layout()

# fig.savefig('/Users/ampuku/Documents/duct/Fig/_paper_figure/f_Ne_f_B_plot4.png', dpi=300)

# %%
labels = ['(b1)', '(c1)', '(b2)', '(c2)', '(b3)', '(c3)', '(b4)', '(c4)']
i = 0
for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]:
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    tx = (xlims[1] - xlims[0]) * 0.03 + xlims[0]
    ty = (ylims[1] - ylims[0]) * 0.84 + ylims[0]
    ax.text(tx, ty, labels[i], fontsize=24)
    i += 1

# %%
# fig.tight_layout()
# fig.savefig('/Users/ampuku/Documents/duct/Fig/_paper_figure/f_Ne_f_B_plot_all_v2.png', dpi=300)

# %%


"""

---------------------------------------------
-------------------以下WNA-------------------
---------------------------------------------

"""


# %%
path = '/Users/ampuku/Documents/duct/code/python/for_paper_figure/f_wna_data1/'

eqfce = pd.read_csv(path+'eqfce_data.csv', header=None)[0]
fce_ave = pd.read_csv(path+'fce_data.csv', header=None)[0]
f_kvec_obs = pd.read_csv(path+'fkvecobs_data.csv', header=None)
gendrinangle = pd.read_csv(path+'gendrinangle_data.csv', header=None)
kvec_obs = pd.read_csv(path+'kvecobs_data.csv', header=None)

eqfce = np.array(eqfce)
fce_ave = np.array(fce_ave)
f_obs = np.array(f_kvec_obs[0])
f_kvec_obs = np.array(f_kvec_obs).reshape((1, -1))[0]
kvec_obs = np.array(kvec_obs).reshape((1, -1))[0]

f_kvec_obs = f_kvec_obs[~np.isnan(kvec_obs)]
kvec_obs = kvec_obs[~np.isnan(kvec_obs)]

# %%
# plt.rcParams["font.size"] = 20
# fig = plt.figure(figsize=(24, 6))

ax1 = fig.add_subplot(2, 4, 1)
ax1.scatter(f_kvec_obs/fce_ave, kvec_obs, color='k')
ax1.plot(f_obs/fce_ave, gendrinangle, color='r', linewidth=3, linestyle='solid')
ax1.set_xlim(0.05, 0.30)
ax1.set_ylim(0, 90)
ax1.set_xlabel("f / fc")
# ax1.set_xticklabels([])
ax1.set_ylabel("wave normal angle\n[degree]")
ax1.vlines(eqfce/fce_ave, -10, 120, colors='k', linestyle='dotted', linewidth=3)

# %%
path = '/Users/ampuku/Documents/duct/code/python/for_paper_figure/f_wna_data2/'

eqfce = pd.read_csv(path+'eqfce_data.csv', header=None)[0]
fce_ave = pd.read_csv(path+'fce_data.csv', header=None)[0]
f_kvec_obs = pd.read_csv(path+'fkvecobs_data.csv', header=None)
gendrinangle = pd.read_csv(path+'gendrinangle_data.csv', header=None)
kvec_obs = pd.read_csv(path+'kvecobs_data.csv', header=None)

eqfce = np.array(eqfce)
fce_ave = np.array(fce_ave)
f_obs = np.array(f_kvec_obs[0])
f_kvec_obs = np.array(f_kvec_obs).reshape((1, -1))[0]
kvec_obs = np.array(kvec_obs).reshape((1, -1))[0]

f_kvec_obs = f_kvec_obs[~np.isnan(kvec_obs)]
kvec_obs = kvec_obs[~np.isnan(kvec_obs)]

# %%
ax2 = fig.add_subplot(2, 4, 2)
ax2.scatter(f_kvec_obs/fce_ave, kvec_obs, color='k')
ax2.plot(f_obs/fce_ave, gendrinangle, color='r', linewidth=3, linestyle='solid')
ax2.set_xlim(0.25, 0.5)
ax2.set_ylim(0, 90)
ax2.set_xlabel("f / fc")
# ax2.set_xticklabels([])
ax2.set_ylabel("wave normal angle\n[degree]")
# ax2.set_yticklabels("")
ax2.vlines(eqfce/fce_ave, -10, 120, colors='k', linestyle='dotted', linewidth=3)

# %%
path = '/Users/ampuku/Documents/duct/code/python/for_paper_figure/f_wna_data3/'

eqfce = pd.read_csv(path+'eqfce_data.csv', header=None)[0]
fce_ave = pd.read_csv(path+'fce_data.csv', header=None)[0]
f_kvec_obs = pd.read_csv(path+'fkvecobs_data.csv', header=None)
gendrinangle = pd.read_csv(path+'gendrinangle_data.csv', header=None)
kvec_obs = pd.read_csv(path+'kvecobs_data.csv', header=None)

eqfce = np.array(eqfce)
fce_ave = np.array(fce_ave)
f_obs = np.array(f_kvec_obs[0])
f_kvec_obs = np.array(f_kvec_obs).reshape((1, -1))[0]
kvec_obs = np.array(kvec_obs).reshape((1, -1))[0]

f_kvec_obs = f_kvec_obs[~np.isnan(kvec_obs)]
kvec_obs = kvec_obs[~np.isnan(kvec_obs)]

# %%
ax3 = fig.add_subplot(2, 4, 3)
ax3.scatter(f_kvec_obs/fce_ave, kvec_obs, color='k')
ax3.plot(f_obs/fce_ave, gendrinangle, color='r', linewidth=3, linestyle='solid')
ax3.set_xlim(0.02, 0.17)
ax3.set_ylim(0, 90)
ax3.set_xlabel("f / fc")
# ax3.set_xticklabels([])
ax3.set_ylabel("wave normal angle\n[degree]")
# ax3.set_yticklabels("")
ax3.vlines(eqfce/fce_ave, -10, 120, colors='k', linestyle='dotted', linewidth=3)

# %%
path = '/Users/ampuku/Documents/duct/code/python/for_paper_figure/f_wna_data4/'

eqfce = pd.read_csv(path+'eqfce_data.csv', header=None)[0]
fce_ave = pd.read_csv(path+'fce_data.csv', header=None)[0]
f_kvec_obs = pd.read_csv(path+'fkvecobs_data.csv', header=None)
gendrinangle = pd.read_csv(path+'gendrinangle_data.csv', header=None)
kvec_obs = pd.read_csv(path+'kvecobs_data.csv', header=None)

# %%
eqfce = np.array(eqfce)
fce_ave = np.array(fce_ave)
f_obs = np.array(f_kvec_obs[0])
f_kvec_obs = np.array(f_kvec_obs).reshape((1, -1))[0]
kvec_obs = np.array(kvec_obs).reshape((1, -1))[0]

f_kvec_obs = f_kvec_obs[~np.isnan(kvec_obs)]
kvec_obs = kvec_obs[~np.isnan(kvec_obs)]

# %%
ax4 = fig.add_subplot(2, 4, 4)
# ax4.scatter(f_kvec_obs/fce_ave, kvec_obs, color='k')
ax4.scatter(f_kvec_obs/fce_ave, kvec_obs, color='k', alpha=0.1)
ax4.plot(f_obs/fce_ave, gendrinangle, color='r', linewidth=3, linestyle='solid')
ax4.set_xlim(0.3, 0.6)
ax4.set_ylim(0, 90)
ax4.set_xlabel("f / fc")
# ax4.set_xlabel("f / fc")
# ax4.set_xticklabels([])
ax4.set_ylabel("wave normal angle\n[degree]")
# ax4.set_yticklabels("")
ax4.vlines(eqfce/fce_ave, -10, 120, colors='k', linestyle='dotted', linewidth=3)

# %%
spines = 2
for ax in [ax1, ax2, ax3, ax4]:
    ax.spines["top"].set_linewidth(spines)
    ax.spines["left"].set_linewidth(spines)
    ax.spines["bottom"].set_linewidth(spines)
    ax.spines["right"].set_linewidth(spines)
    ax.minorticks_on()
    ax.tick_params(which='major', width=2, length=6)
    ax.tick_params(which='minor', width=2, length=3)
    ax.grid()

labels = ['(a1)', '(a2)', '(a3)', '(a4)']
i = 0
for ax in [ax1, ax2, ax3, ax4]:
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    tx = (xlims[1] - xlims[0]) * 0.04 + xlims[0]
    ty = (ylims[1] - ylims[0]) * 0.9 + ylims[0]
    ax.text(tx, ty, labels[i], fontsize=24)
    i += 1


fig.tight_layout()
fig.subplots_adjust(wspace=0.3, hspace=0.35)
# fig.subplots_adjust(bottom=0, left=0, top=1, right=1)
fig.savefig('/Users/ampuku/Desktop/wna_ne_b_plot_all.png', dpi=300)
# fig.savefig('/Users/ampuku/Documents/duct/Fig/_paper_figure/wna_ne_b_plot_all2.png', dpi=300)


# %%
