# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import japanize_matplotlib
import seaborn as sns

# theta_output_all, theta_input, theta_input

def axplots(ax, df_all, x_input, x_expected=None, label=None):
    # sns.violinplot(data=df_all ,inner=None ,palette=['gray'], ax=ax)
    # sns.boxplot(data=df_all ,inner='box' ,palette=['gray'], ax=ax)
    # sns.catplot(data=df_all, kind='boxen')
    ax.errorbar(x_input, df_all.mean(), yerr=df_all.std(), capsize=3, fmt='o', color='w', ecolor='k', ms=7, mec='k')
    # ax.errorbar(x_input, df_all.mean(), yerr=df_all.std(), capsize=3, fmt='o', color='k', ecolor='k', ms=7, mfc='None', mec='k')
    # ax.errorbar((x_input-x_input[0])/(x_input[-1]-x_input[0])*(len(x_input)-1), df_all.mean(), yerr=df_all.std(), capsize=3, fmt='o', ecolor='k', ms=7, mfc='None', mec='k')
    if not x_expected is None:
        ax.plot(x_input, x_expected, color='gray')
        # ax.plot((x_input-x_input[0])/(x_input[-1]-x_input[0])*(len(x_input)-1), x_expected, color='gray')
    # ax.set_xticklabels([5, 15, 25, 35, 45, 55, 65])
    # ax.set_xticklabels(x_input)
    ax.vlines(66.4, -200, 200, color='r')
    # ax.vlines(45.6, -200, 200, color='r')
    ax.set_xticks([0,10,20,30,40,50,60,70,80]) 
    ax.set_xlim([0,80])
    ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)

    # labels = ['（a1）', '（a2）', '（a3）', '（b1）', '（b2）', '（b3）', '（c1）', '（c2）', '（c3）']

    # i = 0
    if not label is None:
        xlims = ax.get_xlim()
        ylims = ax.get_ylim()
        tx = (xlims[1] - xlims[0]) * 0.04 + xlims[0]
        ty = (ylims[1] - ylims[0]) * 0.9 + ylims[0]
        print(tx, ty)
        # ax.text(tx, ty, label, fontsize=24)


    return




# %%
dirc = '/Users/ampuku/Documents/duct/code/python/'


# %%

"""
波の数=8に固定
phi=[0.,45.,90.,135.,180.,225.,270.,315.]の等方な分布
横軸:入力値のWNA,10度刻みで変化させてみる
f/fc=0.2, theta_g=66.4
縦軸:SVD結果のplanarity,WNA,phi

"""

path = dirc + 'for_planarity_fwhm/csv/rand_fconst_'
wna5_phi0360_wave8_phi = pd.read_csv(path+'wna5_phi0360_wave8_ffc02_phi.csv', header=0)  # .transpose()
wna15_phi0360_wave8_phi = pd.read_csv(path+'wna15_phi0360_wave8_ffc02_phi.csv', header=0)  # .transpose()
wna25_phi0360_wave8_phi = pd.read_csv(path+'wna25_phi0360_wave8_ffc02_phi.csv', header=0)  # .transpose()
wna35_phi0360_wave8_phi = pd.read_csv(path+'wna35_phi0360_wave8_ffc02_phi.csv', header=0)  # .transpose()
wna45_phi0360_wave8_phi = pd.read_csv(path+'wna45_phi0360_wave8_ffc02_phi.csv', header=0)  # .transpose()
wna55_phi0360_wave8_phi = pd.read_csv(path+'wna55_phi0360_wave8_ffc02_phi.csv', header=0)  # .transpose()
wna65_phi0360_wave8_phi = pd.read_csv(path+'wna65_phi0360_wave8_ffc02_phi.csv', header=0)  # .transpose()
wna75_phi0360_wave8_phi = pd.read_csv(path+'wna75_phi0360_wave8_ffc02_phi.csv', header=0)  # .transpose()

freq = np.array(list(map(float, wna5_phi0360_wave8_phi.columns.tolist())))

# display(wna5_phi0360_wave8_phi.describe())
names_phi = [wna5_phi0360_wave8_phi, wna15_phi0360_wave8_phi, wna25_phi0360_wave8_phi, wna35_phi0360_wave8_phi,
            #  wna45_phi0360_wave8_phi, wna55_phi0360_wave8_phi, wna65_phi0360_wave8_phi]
             wna45_phi0360_wave8_phi, wna55_phi0360_wave8_phi, wna65_phi0360_wave8_phi, wna75_phi0360_wave8_phi]


path = dirc + 'for_planarity_fwhm/csv/rand_fconst_'
wna5_phi0360_wave8_theta = pd.read_csv(path+'wna5_phi0360_wave8_ffc02_theta.csv', header=0)  # .transpose()
wna15_phi0360_wave8_theta = pd.read_csv(path+'wna15_phi0360_wave8_ffc02_theta.csv', header=0)  # .transpose()
wna25_phi0360_wave8_theta = pd.read_csv(path+'wna25_phi0360_wave8_ffc02_theta.csv', header=0)  # .transpose()
wna35_phi0360_wave8_theta = pd.read_csv(path+'wna35_phi0360_wave8_ffc02_theta.csv', header=0)  # .transpose()
wna45_phi0360_wave8_theta = pd.read_csv(path+'wna45_phi0360_wave8_ffc02_theta.csv', header=0)  # .transpose()
wna55_phi0360_wave8_theta = pd.read_csv(path+'wna55_phi0360_wave8_ffc02_theta.csv', header=0)  # .transpose()
wna65_phi0360_wave8_theta = pd.read_csv(path+'wna65_phi0360_wave8_ffc02_theta.csv', header=0)  # .transpose()
wna75_phi0360_wave8_theta = pd.read_csv(path+'wna75_phi0360_wave8_ffc02_theta.csv', header=0)  # .transpose()

freq = np.array(list(map(float, wna5_phi0360_wave8_theta.columns.tolist())))

# display(wna5_phi0360_wave8_theta.describe())
names_theta = [wna5_phi0360_wave8_theta, wna15_phi0360_wave8_theta, wna25_phi0360_wave8_theta, wna35_phi0360_wave8_theta,
            #    wna45_phi0360_wave8_theta, wna55_phi0360_wave8_theta, wna65_phi0360_wave8_theta]
               wna45_phi0360_wave8_theta, wna55_phi0360_wave8_theta, wna65_phi0360_wave8_theta, wna75_phi0360_wave8_theta]


path = dirc + 'for_planarity_fwhm/csv/rand_fconst_'
wna5_phi0360_wave8_pla = pd.read_csv(path+'wna5_phi0360_wave8_ffc02_pla.csv', header=0)  # .transpose()
wna15_phi0360_wave8_pla = pd.read_csv(path+'wna15_phi0360_wave8_ffc02_pla.csv', header=0)  # .transpose()
wna25_phi0360_wave8_pla = pd.read_csv(path+'wna25_phi0360_wave8_ffc02_pla.csv', header=0)  # .transpose()
wna35_phi0360_wave8_pla = pd.read_csv(path+'wna35_phi0360_wave8_ffc02_pla.csv', header=0)  # .transpose()
wna45_phi0360_wave8_pla = pd.read_csv(path+'wna45_phi0360_wave8_ffc02_pla.csv', header=0)  # .transpose()
wna55_phi0360_wave8_pla = pd.read_csv(path+'wna55_phi0360_wave8_ffc02_pla.csv', header=0)  # .transpose()
wna65_phi0360_wave8_pla = pd.read_csv(path+'wna65_phi0360_wave8_ffc02_pla.csv', header=0)  # .transpose()
wna75_phi0360_wave8_pla = pd.read_csv(path+'wna75_phi0360_wave8_ffc02_pla.csv', header=0)  # .transpose()

freq = np.array(list(map(float, wna5_phi0360_wave8_pla.columns.tolist())))

# display(wna5_phi0360_wave8_pla.describe())
names_pla = [wna5_phi0360_wave8_pla, wna15_phi0360_wave8_pla, wna25_phi0360_wave8_pla, wna35_phi0360_wave8_pla,
            #  wna45_phi0360_wave8_pla, wna55_phi0360_wave8_pla, wna65_phi0360_wave8_pla]
             wna45_phi0360_wave8_pla, wna55_phi0360_wave8_pla, wna65_phi0360_wave8_pla, wna75_phi0360_wave8_pla]


# forcus_f = 1024.0 # [:,16]
forcus_f = 576.0 # [:,9]
theta_input = np.arange(5, 80, 10)
index = np.array([5, 15, 25, 35, 45, 55, 65, 75])
col=np.arange(len(wna65_phi0360_wave8_pla.index), dtype='int')

theta_output_all = []
phi_output_all = []
pla_output_all = []

for i in range(len(theta_input)):
    theta_output_all.append(names_theta[i].iloc[:,8])
    phi_output_all.append(names_phi[i].iloc[:,8])
    pla_output_all.append(names_pla[i].iloc[:,8])


theta_output_all = pd.DataFrame(theta_output_all, index=index, columns=col).T
phi_output_all = pd.DataFrame(phi_output_all, index=index, columns=col).T
pla_output_all = pd.DataFrame(pla_output_all, index=index, columns=col).T


fig = plt.figure(figsize=(15, 15))
plt.rcParams["font.size"] = 18

ax1 = fig.add_subplot(3, 3, 1)
ax1.set_title('phi : 0-360 deg\n', size=20)
# ax1.set_title('num of waves : 8, phi : 0-360 deg\nnfft区間ごとの位相がランダム、周波数変化なし', size=15)
axplots(ax1, theta_output_all, theta_input, theta_input, '（a1）')
# ax1.set_xlabel('input theta [degree]')
ax1.set_ylabel('mag SVD theta [degree]')
ax1.set_ylim([0, 90])
ax1.tick_params(labelbottom=False, labelleft=True)
ax1.grid()

ax2 = fig.add_subplot(3, 3, 4)
axplots(ax2, phi_output_all, theta_input, theta_input*0, '（a2）')
# ax2.set_xlabel('input theta [degree]')
ax2.set_ylabel('mag SVD phi [degree]')
ax2.set_ylim([-180, 180])
ax2.tick_params(labelbottom=False, labelleft=True)
ax2.grid()

ax3 = fig.add_subplot(3, 3, 7)
axplots(ax3, pla_output_all, theta_input)
ax3.set_xlabel('input theta [degree]')
ax3.set_ylabel('mag SVD planarity')
ax3.set_ylim([0.3, 1.0])
ax3.tick_params(labelbottom=True, labelleft=True)
ax3.grid()

# for ax in [ax1, ax2, ax3]:
    # ax.vlines(45.6, 0, 90, colors='red', alpha=0.8)


"""
波の数=8に固定
phi=[0.,10.,20.,30.,40.,50.,60.,70.]の偏った分布
横軸:入力値のWNA,10度刻みで変化させてみる
f/fc=0.2, theta_g=66.4
縦軸:SVD結果のplanarity,WNA,phi

"""

path = dirc + 'for_planarity_fwhm/csv/rand_fconst_'
wna5_phi090_wave8_phi = pd.read_csv(path+'wna5_phi090_wave8_ffc02_phi.csv', header=0)  # .transpose()
wna15_phi090_wave8_phi = pd.read_csv(path+'wna15_phi090_wave8_ffc02_phi.csv', header=0)  # .transpose()
wna25_phi090_wave8_phi = pd.read_csv(path+'wna25_phi090_wave8_ffc02_phi.csv', header=0)  # .transpose()
wna35_phi090_wave8_phi = pd.read_csv(path+'wna35_phi090_wave8_ffc02_phi.csv', header=0)  # .transpose()
wna45_phi090_wave8_phi = pd.read_csv(path+'wna45_phi090_wave8_ffc02_phi.csv', header=0)  # .transpose()
wna55_phi090_wave8_phi = pd.read_csv(path+'wna55_phi090_wave8_ffc02_phi.csv', header=0)  # .transpose()
wna65_phi090_wave8_phi = pd.read_csv(path+'wna65_phi090_wave8_ffc02_phi.csv', header=0)  # .transpose()
wna75_phi090_wave8_phi = pd.read_csv(path+'wna75_phi090_wave8_ffc02_phi.csv', header=0)  # .transpose()

freq = np.array(list(map(float, wna5_phi090_wave8_phi.columns.tolist())))

# display(wna5_phi090_wave8_phi.describe())
names_phi = [wna5_phi090_wave8_phi, wna15_phi090_wave8_phi, wna25_phi090_wave8_phi, wna35_phi090_wave8_phi,
            #  wna45_phi090_wave8_phi, wna55_phi090_wave8_phi, wna65_phi090_wave8_phi]
             wna45_phi090_wave8_phi, wna55_phi090_wave8_phi, wna65_phi090_wave8_phi, wna75_phi090_wave8_phi]


path = dirc + 'for_planarity_fwhm/csv/rand_fconst_'
wna5_phi090_wave8_theta = pd.read_csv(path+'wna5_phi090_wave8_ffc02_theta.csv', header=0)  # .transpose()
wna15_phi090_wave8_theta = pd.read_csv(path+'wna15_phi090_wave8_ffc02_theta.csv', header=0)  # .transpose()
wna25_phi090_wave8_theta = pd.read_csv(path+'wna25_phi090_wave8_ffc02_theta.csv', header=0)  # .transpose()
wna35_phi090_wave8_theta = pd.read_csv(path+'wna35_phi090_wave8_ffc02_theta.csv', header=0)  # .transpose()
wna45_phi090_wave8_theta = pd.read_csv(path+'wna45_phi090_wave8_ffc02_theta.csv', header=0)  # .transpose()
wna55_phi090_wave8_theta = pd.read_csv(path+'wna55_phi090_wave8_ffc02_theta.csv', header=0)  # .transpose()
wna65_phi090_wave8_theta = pd.read_csv(path+'wna65_phi090_wave8_ffc02_theta.csv', header=0)  # .transpose()
wna75_phi090_wave8_theta = pd.read_csv(path+'wna75_phi090_wave8_ffc02_theta.csv', header=0)  # .transpose()

freq = np.array(list(map(float, wna5_phi090_wave8_theta.columns.tolist())))

# display(wna5_phi090_wave8_theta.describe())
names_theta = [wna5_phi090_wave8_theta, wna15_phi090_wave8_theta, wna25_phi090_wave8_theta, wna35_phi090_wave8_theta,
            #    wna45_phi090_wave8_theta, wna55_phi090_wave8_theta, wna65_phi090_wave8_theta]
               wna45_phi090_wave8_theta, wna55_phi090_wave8_theta, wna65_phi090_wave8_theta, wna75_phi090_wave8_theta]


path = dirc + 'for_planarity_fwhm/csv/rand_fconst_'
wna5_phi090_wave8_pla = pd.read_csv(path+'wna5_phi090_wave8_ffc02_pla.csv', header=0)  # .transpose()
wna15_phi090_wave8_pla = pd.read_csv(path+'wna15_phi090_wave8_ffc02_pla.csv', header=0)  # .transpose()
wna25_phi090_wave8_pla = pd.read_csv(path+'wna25_phi090_wave8_ffc02_pla.csv', header=0)  # .transpose()
wna35_phi090_wave8_pla = pd.read_csv(path+'wna35_phi090_wave8_ffc02_pla.csv', header=0)  # .transpose()
wna45_phi090_wave8_pla = pd.read_csv(path+'wna45_phi090_wave8_ffc02_pla.csv', header=0)  # .transpose()
wna55_phi090_wave8_pla = pd.read_csv(path+'wna55_phi090_wave8_ffc02_pla.csv', header=0)  # .transpose()
wna65_phi090_wave8_pla = pd.read_csv(path+'wna65_phi090_wave8_ffc02_pla.csv', header=0)  # .transpose()
wna75_phi090_wave8_pla = pd.read_csv(path+'wna75_phi090_wave8_ffc02_pla.csv', header=0)  # .transpose()

freq = np.array(list(map(float, wna5_phi090_wave8_pla.columns.tolist())))

# display(wna5_phi090_wave8_pla.describe())
names_pla = [wna5_phi090_wave8_pla, wna15_phi090_wave8_pla, wna25_phi090_wave8_pla, wna35_phi090_wave8_pla,
            #  wna45_phi090_wave8_pla, wna55_phi090_wave8_pla, wna65_phi090_wave8_pla]
             wna45_phi090_wave8_pla, wna55_phi090_wave8_pla, wna65_phi090_wave8_pla, wna75_phi090_wave8_pla]


# forcus_f = 1024.0
# forcus_f = 576.0
theta_input = np.arange(5, 80, 10, dtype='int')
index = np.array([5, 15, 25, 35, 45, 55, 65,75])
col=np.arange(len(wna65_phi0360_wave8_pla.index), dtype='int')

theta_output_all = []
phi_output_all = []
pla_output_all = []


for i in range(len(theta_input)):
    theta_output_all.append(names_theta[i].iloc[:,9])
    phi_output_all.append(names_phi[i].iloc[:,9])
    pla_output_all.append(names_pla[i].iloc[:,9])

theta_output_all = pd.DataFrame(theta_output_all, index=index, columns=col).T
phi_output_all = pd.DataFrame(phi_output_all, index=index, columns=col).T
pla_output_all = pd.DataFrame(pla_output_all, index=index, columns=col).T


ax1 = fig.add_subplot(3, 3, 2)
ax1.set_title('phi : 0-70 deg\n', size=20)
# ax1.set_title('num of waves : 8, phi : 0-70 deg\nnfft区間ごとの位相がランダム、周波数変化なし', size=15)
axplots(ax1, theta_output_all, theta_input, theta_input, '（b1）')
# ax1.set_xlabel('input theta [degree]')
# ax1.set_ylabel('mag SVD theta [degree]')
ax1.set_ylim([0, 90])
ax1.grid()

ax2 = fig.add_subplot(3, 3, 5)
axplots(ax2, phi_output_all, theta_input, theta_input*0+35, '（b2）')
# ax2.set_xlabel('input theta [degree]')
# ax2.set_ylabel('mag SVD phi [degree]')
ax2.set_ylim([-180, 180])
ax2.grid()

ax3 = fig.add_subplot(3, 3, 8)
axplots(ax3, pla_output_all, theta_input)
ax3.set_xlabel('input theta [degree]')
# ax3.set_ylabel('mag SVD planarity')
ax3.set_ylim([0.3, 1.0])
ax3.tick_params(labelbottom=True, labelleft=False)
ax3.grid()

# plt.tight_layout()
# plt.savefig('/Users/ampuku/Documents/duct/Fig/calcs/rand_fconst_wna575_phi090_wave8')

# for ax in [ax1, ax2, ax3]:
    # ax.vlines(45.6, 0, 90, colors='red', alpha=0.8)






"""
波の数=8に固定
phi=[0.,0.,0.,0.,0.,0.,0.,0.]の等方な分布
横軸:入力値のWNA,10度刻みで変化させてみる
f/fc=0.2, theta_g=66.4
縦軸:SVD結果のplanarity,WNA,phi

"""

path = dirc + 'for_planarity_fwhm/csv/rand_fconst_'
wna5_phi0_wave8_phi = pd.read_csv(path+'wna5_phi0_wave8_ffc02_phi.csv', header=0)  # .transpose()
wna15_phi0_wave8_phi = pd.read_csv(path+'wna15_phi0_wave8_ffc02_phi.csv', header=0)  # .transpose()
wna25_phi0_wave8_phi = pd.read_csv(path+'wna25_phi0_wave8_ffc02_phi.csv', header=0)  # .transpose()
wna35_phi0_wave8_phi = pd.read_csv(path+'wna35_phi0_wave8_ffc02_phi.csv', header=0)  # .transpose()
wna45_phi0_wave8_phi = pd.read_csv(path+'wna45_phi0_wave8_ffc02_phi.csv', header=0)  # .transpose()
wna55_phi0_wave8_phi = pd.read_csv(path+'wna55_phi0_wave8_ffc02_phi.csv', header=0)  # .transpose()
wna65_phi0_wave8_phi = pd.read_csv(path+'wna65_phi0_wave8_ffc02_phi.csv', header=0)  # .transpose()
wna75_phi0_wave8_phi = pd.read_csv(path+'wna75_phi0_wave8_ffc02_phi.csv', header=0)  # .transpose()

freq = np.array(list(map(float, wna5_phi0_wave8_phi.columns.tolist())))

# display(wna5_phi0_wave8_phi.describe())
names_phi = [wna5_phi0_wave8_phi, wna15_phi0_wave8_phi, wna25_phi0_wave8_phi, wna35_phi0_wave8_phi,
            #  wna45_phi0_wave8_phi, wna55_phi0_wave8_phi, wna65_phi0_wave8_phi]
             wna45_phi0_wave8_phi, wna55_phi0_wave8_phi, wna65_phi0_wave8_phi, wna75_phi0_wave8_phi]


path = dirc + 'for_planarity_fwhm/csv/rand_fconst_'
wna5_phi0_wave8_theta = pd.read_csv(path+'wna5_phi0_wave8_ffc02_theta.csv', header=0)  # .transpose()
wna15_phi0_wave8_theta = pd.read_csv(path+'wna15_phi0_wave8_ffc02_theta.csv', header=0)  # .transpose()
wna25_phi0_wave8_theta = pd.read_csv(path+'wna25_phi0_wave8_ffc02_theta.csv', header=0)  # .transpose()
wna35_phi0_wave8_theta = pd.read_csv(path+'wna35_phi0_wave8_ffc02_theta.csv', header=0)  # .transpose()
wna45_phi0_wave8_theta = pd.read_csv(path+'wna45_phi0_wave8_ffc02_theta.csv', header=0)  # .transpose()
wna55_phi0_wave8_theta = pd.read_csv(path+'wna55_phi0_wave8_ffc02_theta.csv', header=0)  # .transpose()
wna65_phi0_wave8_theta = pd.read_csv(path+'wna65_phi0_wave8_ffc02_theta.csv', header=0)  # .transpose()
wna75_phi0_wave8_theta = pd.read_csv(path+'wna75_phi0_wave8_ffc02_theta.csv', header=0)  # .transpose()

freq = np.array(list(map(float, wna5_phi0_wave8_theta.columns.tolist())))

# display(wna5_phi0_wave8_theta.describe())
names_theta = [wna5_phi0_wave8_theta, wna15_phi0_wave8_theta, wna25_phi0_wave8_theta, wna35_phi0_wave8_theta,
            #    wna45_phi0_wave8_theta, wna55_phi0_wave8_theta, wna65_phi0_wave8_theta]
               wna45_phi0_wave8_theta, wna55_phi0_wave8_theta, wna65_phi0_wave8_theta, wna75_phi0_wave8_theta]


path = dirc + 'for_planarity_fwhm/csv/rand_fconst_'
wna5_phi0_wave8_pla = pd.read_csv(path+'wna5_phi0_wave8_ffc02_pla.csv', header=0)  # .transpose()
wna15_phi0_wave8_pla = pd.read_csv(path+'wna15_phi0_wave8_ffc02_pla.csv', header=0)  # .transpose()
wna25_phi0_wave8_pla = pd.read_csv(path+'wna25_phi0_wave8_ffc02_pla.csv', header=0)  # .transpose()
wna35_phi0_wave8_pla = pd.read_csv(path+'wna35_phi0_wave8_ffc02_pla.csv', header=0)  # .transpose()
wna45_phi0_wave8_pla = pd.read_csv(path+'wna45_phi0_wave8_ffc02_pla.csv', header=0)  # .transpose()
wna55_phi0_wave8_pla = pd.read_csv(path+'wna55_phi0_wave8_ffc02_pla.csv', header=0)  # .transpose()
wna65_phi0_wave8_pla = pd.read_csv(path+'wna65_phi0_wave8_ffc02_pla.csv', header=0)  # .transpose()
wna75_phi0_wave8_pla = pd.read_csv(path+'wna75_phi0_wave8_ffc02_pla.csv', header=0)  # .transpose()

freq = np.array(list(map(float, wna5_phi0_wave8_pla.columns.tolist())))

# display(wna5_phi0_wave8_pla.describe())
names_pla = [wna5_phi0_wave8_pla, wna15_phi0_wave8_pla, wna25_phi0_wave8_pla, wna35_phi0_wave8_pla,
            #  wna45_phi0_wave8_pla, wna55_phi0_wave8_pla, wna65_phi0_wave8_pla]
             wna45_phi0_wave8_pla, wna55_phi0_wave8_pla, wna65_phi0_wave8_pla, wna75_phi0_wave8_pla]


# forcus_f = 1024.0
# forcus_f = 576.0
theta_input = np.arange(5, 80, 10, dtype='int')
index = np.array([5, 15, 25, 35, 45, 55, 65, 75])
col=np.arange(len(wna65_phi0360_wave8_pla.index), dtype='int')

theta_output_all = []
phi_output_all = []
pla_output_all = []

for i in range(len(theta_input)):
    theta_output_all.append(names_theta[i].iloc[:,9])
    phi_output_all.append(names_phi[i].iloc[:,9])
    pla_output_all.append(names_pla[i].iloc[:,9])

theta_output_all = pd.DataFrame(theta_output_all, index=index, columns=col).T
phi_output_all = pd.DataFrame(phi_output_all, index=index, columns=col).T
pla_output_all = pd.DataFrame(pla_output_all, index=index, columns=col).T

ax1 = fig.add_subplot(3, 3, 3)
ax1.set_title('phi : 0 deg\n', size=20)
# ax1.set_title('num of waves : 8, phi : 0 deg\nnfft区間ごとの位相がランダム、周波数変化なし', size=15)
axplots(ax1, theta_output_all, theta_input, theta_input, '（c1）')
# ax1.set_xlabel('input theta [degree]')
# ax1.set_ylabel('mag SVD theta [degree]')
ax1.set_ylim([0, 90])
ax1.grid()

ax2 = fig.add_subplot(3, 3, 6)
axplots(ax2, phi_output_all, theta_input, theta_input*0, '（c2）')
# ax2.set_xlabel('input theta [degree]')
# ax2.set_ylabel('mag SVD phi [degree]')
ax2.set_ylim([-180, 180])
ax2.grid()

ax3 = fig.add_subplot(3, 3, 9)
axplots(ax3, pla_output_all, theta_input)
ax3.set_xlabel('input theta [degree]')
# ax3.set_ylabel('mag SVD planarity')
ax3.set_ylim([0.3, 1.0])
ax3.tick_params(labelbottom=True, labelleft=False)
ax3.grid()


plt.tight_layout()
# plt.savefig('/Users/ampuku/Documents/duct/Fig/calcs/rand_fconst_wna575_wave8_withv')
plt.savefig('/Users/ampuku/Documents/duct/Fig/calcs/rand_fconst_wna575_wave8_20_v2')

# %%

# %%
