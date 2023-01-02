# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
plt.rcParams["font.size"] = 20
fig = plt.figure(figsize=(24, 6))

ax1 = fig.add_subplot(1, 4, 1)
ax1.scatter(f_kvec_obs/fce_ave, kvec_obs, color='k')
ax1.plot(f_obs/fce_ave, gendrinangle, color='r', linewidth=3, linestyle='solid')
ax1.set_xlim(0.05, 0.35)
ax1.set_ylim(0, 100)
ax1.set_xlabel("f / fc")
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
ax2 = fig.add_subplot(1, 4, 2)
ax2.scatter(f_kvec_obs/fce_ave, kvec_obs, color='k')
ax2.plot(f_obs/fce_ave, gendrinangle, color='r', linewidth=3, linestyle='solid')
ax2.set_xlim(0.2, 0.5)
ax2.set_ylim(0, 100)
ax2.set_xlabel("f / fc")
ax2.set_ylabel("")
ax2.set_yticklabels("")
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
ax3 = fig.add_subplot(1, 4, 3)
ax3.scatter(f_kvec_obs/fce_ave, kvec_obs, color='k')
ax3.plot(f_obs/fce_ave, gendrinangle, color='r', linewidth=3, linestyle='solid')
ax3.set_xlim(0.0, 0.3)
ax3.set_ylim(0, 100)
ax3.set_xlabel("f / fc")
ax3.set_ylabel("")
ax3.set_yticklabels("")
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
ax4 = fig.add_subplot(1, 4, 4)
ax4.scatter(f_kvec_obs/fce_ave, kvec_obs, color='k')
# ax4.scatter(f_kvec_obs/fce_ave,kvec_obs, color='k', alpha=0.1)
ax4.plot(f_obs/fce_ave, gendrinangle, color='r', linewidth=3, linestyle='solid')
ax4.set_xlim(0.3, 0.6)
ax4.set_ylim(0, 100)
ax4.set_xlabel("f / fc")
ax4.set_ylabel("")
ax4.set_yticklabels("")
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

labels = ['(a)','(b)','(c)','(d)']
i=0
for ax in [ax1,ax2,ax3,ax4]:
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    tx = (xlims[1] - xlims[0]) * 0.04 + xlims[0]
    ty = (ylims[1] - ylims[0]) * 0.9 + ylims[0]
    ax.text(tx, ty, labels[i], fontsize=24)
    i+=1


fig.tight_layout()
fig.savefig('/Users/ampuku/Documents/duct/Fig/_paper_figure/wna_plot_all.png', dpi=300)


# %%
