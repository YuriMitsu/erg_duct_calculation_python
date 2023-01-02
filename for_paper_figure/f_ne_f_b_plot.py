# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %%
fce_ave_all = [14.105383, 13.445186, 30.402452, 14.299986]  # あとで規格化バージョンを上げるか決めたい

# %%
plt.rcParams["font.size"] = 20
fig = plt.figure(figsize=(30, 7))


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
# fce_ave = fce_ave[0]

deff = abs(Ne0 - Nmax)
vlines = [plotf[deff==sorted(deff)[0]], plotf[deff==sorted(deff)[1]]]

# %%

ax1 = fig.add_subplot(2,4,1)
ax1.plot(plotf,Ne0, color='k', linewidth=2)
ax1.plot(plotf,Ne1, color='k', linewidth=2, linestyle='solid')
ax1.axhspan(Nmin, Nmax, color="r", alpha=0.3)
ax1.set_xlim(1.0, 4.0)
ax1.set_ylim(270, 320)
ax1.set_xlabel("")
ax1.set_ylabel("Ne [/cc]")
# ax1.set_xticks([1.,1.5,2.,2.5,3.,3.5,4.])
ax1.set_xticklabels([])



ax2 = fig.add_subplot(2,4,5)
ax2.plot(Bv,Bobs, color='k', linewidth=2)
ax2.set_xlim(1.0, 4.0)
ax2.set_ylim(0., 0.1)
ax2.set_xlabel("frequency [kHz]")
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
    ax1.vlines(xx, 270, 340, colors='k', linestyle='solid', linewidth=2)
    ax2.vlines(xx, -0.1, 0.2, colors='k', linestyle='solid', linewidth=2)
ax2.vlines(fce_eq, -0.1, 0.2, colors='k', linestyle='dotted', linewidth=2)

fig.tight_layout()

fig.savefig('/Users/ampuku/Documents/duct/Fig/_paper_figure/f_Ne_f_B_plot1.png', dpi=300)

















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

deff = abs(Ne0 - Nmin)
vlines1 = [plotf[deff==sorted(deff)[0]], plotf[deff==sorted(deff)[1]]]
deff = abs(Ne1 - Nmax)
vlines2 = [plotf[deff==sorted(deff)[0]]]

# %%
plt.rcParams["font.size"] = 20
fig = plt.figure(figsize=(12, 10))

# ax3 = fig.add_subplot(2,4,2)
ax3 = fig.add_subplot(2,1,1)
ax3.plot(plotf,Ne0, color='k', linewidth=2)
ax3.plot(plotf,Ne1, color='k', linewidth=2, linestyle='dashed')
ax3.axhspan(Nmin, Nmax, color="b", alpha=0.3)
ax3.set_xlim(3.5, 6.5)
ax3.set_ylim(200, 280)
ax3.set_xlabel("")
ax3.set_ylabel("Ne [/cc]")
# ax3.set_xticks([1.,1.5,2.,2.5,3.,3.5,4.])
ax3.set_xticklabels([])



# ax4 = fig.add_subplot(2,4,6)
ax4 = fig.add_subplot(2,1,2)
ax4.plot(Bv,Bobs, color='k', linewidth=2)
ax4.set_xlim(3.5, 6.5)
ax4.set_ylim(0., 0.08)
ax4.set_xlabel("frequency [kHz]")
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
    ax3.vlines(xx, 180, 340, colors='k', linestyle='solid', linewidth=2)
    ax4.vlines(xx, -0.1, 0.2, colors='k', linestyle='solid', linewidth=2)
for xx in vlines2:
    ax3.vlines(xx, 180, 340, colors='k', linestyle='dashed', linewidth=2)
    ax4.vlines(xx, -0.1, 0.2, colors='k', linestyle='dashed', linewidth=2)
ax4.vlines(fce_eq, -0.1, 0.2, colors='k', linestyle='dotted', linewidth=2)

fig.tight_layout()

fig.savefig('/Users/ampuku/Documents/duct/Fig/_paper_figure/f_Ne_f_B_plot2.png', dpi=300)










# %%

path = '/Users/ampuku/Documents/duct/code/python/for_paper_figure/f_Ne_f_B_data3/'
Nmin, Nmax = 128., 166.
# Nmin, Nmax = 128., 230.

Bdata = pd.read_csv(path+'Bv_Bobs_data.csv', header=None)
Bv = Bdata[0]
Bobs = Bdata[1]
Ndata = pd.read_csv(path+'plotf_Ne0_Ne1_data.csv', header=None)
plotf = Ndata[0]
Ne0 = Ndata[1]
Ne1 = Ndata[2]
Fdata = pd.read_csv(path+'eqfce_data.csv', header=None)
fce_eq = Fdata[0]

deff1 = np.array(plotf[abs(Ne1-Nmin)==sorted(abs(Ne1-Nmin))[0]])
deff2 = np.array(plotf[abs(Ne1-Nmax)==sorted(abs(Ne1-Nmax))[0]])
vlines = [deff1[0], deff2[0]]

# %%
plt.rcParams["font.size"] = 20
fig = plt.figure(figsize=(12, 10))

# ax5 = fig.add_subplot(2,4,3)
ax5 = fig.add_subplot(2,1,1)
ax5.plot(plotf,Ne0, color='k', linewidth=2)
ax5.plot(plotf,Ne1, color='k', linewidth=2, linestyle='dashed')
ax5.axhspan(Nmin, Nmax, color="b", alpha=0.3)
ax5.set_xlim(1.5, 4.5)
ax5.set_ylim(100, 190)
# ax5.set_ylim(100, 250)
ax5.set_xlabel("")
ax5.set_ylabel("Ne [/cc]")
# ax5.set_xticks([1.,1.5,2.,2.5,3.,3.5,4.])
ax5.set_xticklabels([])



# ax6 = fig.add_subplot(2,4,7)
ax6 = fig.add_subplot(2,1,2)
ax6.plot(Bv,Bobs, color='k', linewidth=2)
ax6.set_xlim(1.5, 4.5)
ax6.set_ylim(0., 0.05)
ax6.set_xlabel("frequency [kHz]")
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
    ax5.vlines(xx, 90, 300, colors='k', linestyle='dashed', linewidth=2)
    ax6.vlines(xx, -0.1, 0.2, colors='k', linestyle='dashed', linewidth=2)
ax6.vlines(fce_eq, -0.1, 0.2, colors='k', linestyle='dotted', linewidth=2)

fig.tight_layout()

fig.savefig('/Users/ampuku/Documents/duct/Fig/_paper_figure/f_Ne_f_B_plot3.png', dpi=300)

# %%














# %%

path = '/Users/ampuku/Documents/duct/code/python/for_paper_figure/f_Ne_f_B_data4/'
# 最小UHR　：　139.1-140.3 kHz → 誤差計算結果 : 237.5-241.7 ± 4.21-4.25 /cc
# 最大UHR　：　142.6-144.0, 145.3 kHz → 誤差計算結果 : 249.8-254.7, 259.4 ± 4.32-4.36, 4.40 /cc
# Nmin, Nmax = 242., 250.
# Nmin, Nmax = 242., 259.4+4.4 # ちょっと無理ある
Nmin, Nmax = 242.-4.2, 256.5+4.4 # 素直に読むとこれ
# Nmin_err, Nmax_err = 242.-4.2, 250.+4.3

Bdata = pd.read_csv(path+'Bv_Bobs_data.csv', header=None)
Bv = Bdata[0]
Bobs = Bdata[1]
Ndata = pd.read_csv(path+'plotf_Ne0_Ne1_data.csv', header=None)
plotf = Ndata[0]
# Ne01 = np.array(Ndata[1])
# Ne11 = np.array(Ndata[2])
Fdata = pd.read_csv(path+'eqfce_data.csv', header=None)
fce_eq = Fdata[0]
# fce_eq1 = Fdata[0]


# path = '/Users/ampuku/Documents/duct/code/python/for_paper_figure/f_Ne_f_B_data42/'

# Bdata = pd.read_csv(path+'Bv_Bobs_data.csv', header=None)
# Bv2 = Bdata[0]
# Bobs2 = Bdata[1]
# Ndata = pd.read_csv(path+'plotf_Ne0_Ne1_data.csv', header=None)
# plotf2 = Ndata[0]
Ne0 = np.array(Ndata[1])
Ne1 = np.array(Ndata[2])
# Fdata = pd.read_csv(path+'eqfce_data.csv', header=None)
# fce_eq2 = Fdata[0]


# %%
# Ne0 = np.concatenate([Ne01[plotf < fce_ave_all[3]/2], Ne02[plotf > fce_ave_all[3]/2]])
# Ne1 = np.concatenate([Ne11[plotf < fce_ave_all[3]/2], Ne12[plotf > fce_ave_all[3]/2]])


# %%

deff1 = abs(Ne0 - Nmin)
vlines_s = [plotf[deff1==sorted(deff1)[0]], plotf[deff1==sorted(deff1)[1]]]
deff2 = np.array(plotf[abs(Ne1-Nmax)==sorted(abs(Ne1-Nmax))[0]])
vlines_d = [deff2[0]]


# %%
plt.rcParams["font.size"] = 20
fig = plt.figure(figsize=(12, 10))

ax7 = fig.add_subplot(2,1,1)
ax7.plot(plotf,Ne0, color='k', linewidth=2)
ax7.plot(plotf,Ne1, color='k', linewidth=2, linestyle='dashed')
ax7.axhspan(Nmin, Nmax, color="b", alpha=0.3)
ax7.set_xlim(4., 9)
ax7.set_ylim(220, 280)
ax7.set_xlabel("")
ax7.set_ylabel("Ne [/cc]")
# ax7.set_xticks([1.,1.5,2.,2.5,3.,3.5,4.])
ax7.set_xticklabels([])



ax8 = fig.add_subplot(2,1,2)
ax8.plot(Bv,Bobs, color='k', linewidth=2)
ax8.set_xlim(4., 9)
ax8.set_ylim(0., 0.11)
ax8.set_xlabel("frequency [kHz]")
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
    ax7.vlines(xx, 100, 300, colors='k', linestyle='solid', linewidth=2)
    ax8.vlines(xx, -0.1, 0.2, colors='k', linestyle='solid', linewidth=2)
for xx in vlines_d:
    ax7.vlines(xx, 100, 300, colors='k', linestyle='dashed', linewidth=2)
    ax8.vlines(xx, -0.1, 0.2, colors='k', linestyle='dashed', linewidth=2)
ax8.vlines(fce_eq, -0.1, 0.2, colors='k', linestyle='dotted', linewidth=2)

# fig.tight_layout()

# fig.savefig('/Users/ampuku/Documents/duct/Fig/_paper_figure/f_Ne_f_B_plot4.png', dpi=300)

# %%
labels = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)']
i=0
for ax in [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8]:
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    tx = (xlims[1] - xlims[0]) * 0.03 + xlims[0]
    ty = (ylims[1] - ylims[0]) * 0.84 + ylims[0]
    ax.text(tx, ty, labels[i], fontsize=24)
    i+=1

# %%
fig.tight_layout()
fig.savefig('/Users/ampuku/Documents/duct/Fig/_paper_figure/f_Ne_f_B_plot_all.png', dpi=300)

# %%
