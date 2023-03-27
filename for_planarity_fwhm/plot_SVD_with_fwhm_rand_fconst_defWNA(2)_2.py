#

# M論の5_1作成に使用


#  %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import japanize_matplotlib
import seaborn as sns


def axplots(ax, df_all, x_input, x_expected=None):
    # sns.violinplot(data=df_all ,inner='box' ,palette=['gray'], ax=ax)
    # sns.boxplot(data=df_all, palette=['gray'], ax=ax)
    ax.errorbar((x_input-x_input[0])/(x_input[-1]-x_input[0])*(len(x_input)-1), pd.DataFrame(df_all).T.mean(),
                yerr=pd.DataFrame(df_all).T.std(), capsize=3, fmt='o', color='w', ecolor='k', ms=7, mec='k')
    # ax.errorbar((x_input-x_input[0])/(x_input[-1]-x_input[0])*(len(x_input)-1), pd.DataFrame(df_all).T.mean(), yerr=pd.DataFrame(df_all).T.std(), capsize=3, fmt='o', ecolor='k', ms=7, mfc='None', mec='k')
    if not x_expected is None:
        ax.plot((x_input-x_input[0])/(x_input[-1]-x_input[0])*(len(x_input)-1), x_expected, color='gray')
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xlim([-0.5, 3.5])
    ax.set_xticklabels(['45', '40to50', '25to55', '0to60'], fontsize=18)
    ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    return


# %%

"""
波の数=4*n
WNAごとにphi=[0.,90.,180.,270.]の等方な分布
横軸:入力値のWNAの幅が広がっていく
f/fc=0.35, theta_g=45.6
縦軸:SVD結果のplanarity,WNA,phi

"""

path = '/Users/ampuku/Documents/duct/code/python/for_planarity_fwhm/csv/rand_fconst_'
wna45_phi0360_wave4_phi = pd.read_csv(path+'wna45_phi0360_wave4_phi.csv', header=0)  # .transpose()
wna40to50_phi0360_wave12_phi = pd.read_csv(path+'wna40to50_phi0360_wave12_phi.csv', header=0)  # .transpose()
wna25to55_phi0360_wave28_phi = pd.read_csv(path+'wna25to55_phi0360_wave28_phi.csv', header=0)  # .transpose()
wna0to60_phi0360_wave52_phi = pd.read_csv(path+'wna0to60_phi0360_wave52_phi.csv', header=0)  # .transpose()


freq = np.array(list(map(float, wna45_phi0360_wave4_phi.columns.tolist())))

# display(wna45_phi0360_wave4_phi.describe())
names_phi = [wna45_phi0360_wave4_phi, wna40to50_phi0360_wave12_phi, wna25to55_phi0360_wave28_phi, wna0to60_phi0360_wave52_phi]


path = '/Users/ampuku/Documents/duct/code/python/for_planarity_fwhm/csv/rand_fconst_'
wna45_phi0360_wave4_theta = pd.read_csv(path+'wna45_phi0360_wave4_theta.csv', header=0)  # .transpose()
wna40to50_phi0360_wave12_theta = pd.read_csv(path+'wna40to50_phi0360_wave12_theta.csv', header=0)  # .transpose()
wna25to55_phi0360_wave28_theta = pd.read_csv(path+'wna25to55_phi0360_wave28_theta.csv', header=0)  # .transpose()
wna0to60_phi0360_wave52_theta = pd.read_csv(path+'wna0to60_phi0360_wave52_theta.csv', header=0)  # .transpose()


freq = np.array(list(map(float, wna45_phi0360_wave4_theta.columns.tolist())))

# display(wna45_phi0360_wave4_theta.describe())
names_theta = [wna45_phi0360_wave4_theta, wna40to50_phi0360_wave12_theta, wna25to55_phi0360_wave28_theta, wna0to60_phi0360_wave52_theta]


path = '/Users/ampuku/Documents/duct/code/python/for_planarity_fwhm/csv/rand_fconst_'
wna45_phi0360_wave4_pla = pd.read_csv(path+'wna45_phi0360_wave4_pla.csv', header=0)  # .transpose()
wna40to50_phi0360_wave12_pla = pd.read_csv(path+'wna40to50_phi0360_wave12_pla.csv', header=0)  # .transpose()
wna25to55_phi0360_wave28_pla = pd.read_csv(path+'wna25to55_phi0360_wave28_pla.csv', header=0)  # .transpose()
wna0to60_phi0360_wave52_pla = pd.read_csv(path+'wna0to60_phi0360_wave52_pla.csv', header=0)  # .transpose()


freq = np.array(list(map(float, wna45_phi0360_wave4_pla.columns.tolist())))

# display(wna45_phi0360_wave4_pla.describe())
names_pla = [wna45_phi0360_wave4_pla, wna40to50_phi0360_wave12_pla, wna25to55_phi0360_wave28_pla, wna0to60_phi0360_wave52_pla]


forcus_f = 1024.0
theta_input = np.arange(5., 40., 10.)
theta_input_ave = (np.array([45., 40., 25., 0.])+np.array([45., 50., 55., 60.])) / 2
theta_output_all = []
phi_output_all = []
pla_output_all = []

for i in range(len(theta_input)):
    theta_output_all.append(names_theta[i].iloc[:, 16])
    phi_output_all.append(names_phi[i].iloc[:, 16])
    pla_output_all.append(names_pla[i].iloc[:, 16])


# fig = plt.figure(figsize=(5, 15))
fig = plt.figure(figsize=(15, 15))
plt.rcParams["font.size"] = 18
# fig.suptitle('nfft区間ごとの位相がランダム、周波数変化なし    複数WNAを足し合わせた場合', size=20)
# ax1 = fig.add_subplot(3, 1, 1)
ax1 = fig.add_subplot(3, 3, 1)
ax1.set_title('phi : 0-360 deg\n', size=20)
axplots(ax1, theta_output_all, theta_input, theta_input_ave)
# ax1.set_xlabel('input theta [degree]')
ax1.set_ylabel('mag SVD theta [degree]')
ax1.set_ylim([0, 90])
ax1.tick_params(labelbottom=False, labelleft=True)
ax1.grid()

# ax2 = fig.add_subplot(3, 1, 2)
ax2 = fig.add_subplot(3, 3, 4)
axplots(ax2, phi_output_all, theta_input, theta_input*0)
# ax2.set_xlabel('input theta [degree]')
ax2.set_ylabel('mag SVD phi [degree]')
ax2.set_ylim([-180, 180])
ax2.tick_params(labelbottom=False, labelleft=True)
ax2.grid()
# ax3 = fig.add_subplot(3, 1, 3)
ax3 = fig.add_subplot(3, 3, 7)
axplots(ax3, pla_output_all, theta_input)
ax3.set_xlabel('input theta [degree]')
ax3.set_ylabel('mag SVD planarity')
ax3.set_ylim([0.3, 1.0])
ax3.grid()
ax3.tick_params(labelbottom=True, labelleft=True)


"""
波の数=4*n
WNAごとにphi=[0.,20.,40.,60.]の偏った分布
横軸:入力値のWNAの幅が広がっていく
f/fc=0.35, theta_g=45.6
縦軸:SVD結果のplanarity,WNA,phi

"""


path = '/Users/ampuku/Documents/duct/code/python/for_planarity_fwhm/csv/rand_fconst_'
wna45_phi060_wave4_phi = pd.read_csv(path+'wna45_phi060_wave4_phi.csv', header=0)  # .transpose()
wna40to50_phi060_wave12_phi = pd.read_csv(path+'wna40to50_phi060_wave12_phi.csv', header=0)  # .transpose()
wna25to55_phi060_wave28_phi = pd.read_csv(path+'wna25to55_phi060_wave28_phi.csv', header=0)  # .transpose()
wna0to60_phi060_wave52_phi = pd.read_csv(path+'wna0to60_phi060_wave52_phi.csv', header=0)  # .transpose()


freq = np.array(list(map(float, wna45_phi060_wave4_phi.columns.tolist())))

# display(wna45_phi060_wave4_phi.describe())
names_phi = [wna45_phi060_wave4_phi, wna40to50_phi060_wave12_phi, wna25to55_phi060_wave28_phi, wna0to60_phi060_wave52_phi]


path = '/Users/ampuku/Documents/duct/code/python/for_planarity_fwhm/csv/rand_fconst_'
wna45_phi060_wave4_theta = pd.read_csv(path+'wna45_phi060_wave4_theta.csv', header=0)  # .transpose()
wna40to50_phi060_wave12_theta = pd.read_csv(path+'wna40to50_phi060_wave12_theta.csv', header=0)  # .transpose()
wna25to55_phi060_wave28_theta = pd.read_csv(path+'wna25to55_phi060_wave28_theta.csv', header=0)  # .transpose()
wna0to60_phi060_wave52_theta = pd.read_csv(path+'wna0to60_phi060_wave52_theta.csv', header=0)  # .transpose()


freq = np.array(list(map(float, wna45_phi060_wave4_theta.columns.tolist())))

# display(wna45_phi060_wave4_theta.describe())
names_theta = [wna45_phi060_wave4_theta, wna40to50_phi060_wave12_theta, wna25to55_phi060_wave28_theta, wna0to60_phi060_wave52_theta]


path = '/Users/ampuku/Documents/duct/code/python/for_planarity_fwhm/csv/rand_fconst_'
wna45_phi060_wave4_pla = pd.read_csv(path+'wna45_phi060_wave4_pla.csv', header=0)  # .transpose()
wna40to50_phi060_wave12_pla = pd.read_csv(path+'wna40to50_phi060_wave12_pla.csv', header=0)  # .transpose()
wna25to55_phi060_wave28_pla = pd.read_csv(path+'wna25to55_phi060_wave28_pla.csv', header=0)  # .transpose()
wna0to60_phi060_wave52_pla = pd.read_csv(path+'wna0to60_phi060_wave52_pla.csv', header=0)  # .transpose()


freq = np.array(list(map(float, wna45_phi060_wave4_pla.columns.tolist())))

# display(wna45_phi060_wave4_pla.describe())
names_pla = [wna45_phi060_wave4_pla, wna40to50_phi060_wave12_pla, wna25to55_phi060_wave28_pla, wna0to60_phi060_wave52_pla]


forcus_f = 1024.0
theta_input = np.arange(5., 40., 10.)
theta_input_ave = (np.array([45., 40., 25., 0.])+np.array([45., 50., 55., 60.])) / 2
theta_output_all = []
phi_output_all = []
pla_output_all = []

for i in range(len(theta_input)):
    theta_output_all.append(names_theta[i].iloc[:, 16])
    phi_output_all.append(names_phi[i].iloc[:, 16])
    pla_output_all.append(names_pla[i].iloc[:, 16])


# fig = plt.figure(figsize=(5, 15))
ax1 = fig.add_subplot(3, 3, 2)
ax1.set_title('phi : 0-60 deg\n', size=20)
axplots(ax1, theta_output_all, theta_input, theta_input_ave)
# ax1.set_xlabel('input theta [degree]')
# ax1.set_ylabel('mag SVD theta [degree]')
ax1.set_ylim([0, 90])
ax1.grid()
ax2 = fig.add_subplot(3, 3, 5)
axplots(ax2, phi_output_all, theta_input, theta_input*0+30)
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


"""
波の数=4*n
WNAごとにphi=[0.,0.,0.,0.]の等方な分布
横軸:入力値のWNAの幅が広がっていく
f/fc=0.35, theta_g=45.6
縦軸:SVD結果のplanarity,WNA,phi

"""

path = '/Users/ampuku/Documents/duct/code/python/for_planarity_fwhm/csv/rand_fconst_'
wna45_phi0_wave4_phi = pd.read_csv(path+'wna45_phi0_wave4_phi.csv', header=0)  # .transpose()
wna40to50_phi0_wave12_phi = pd.read_csv(path+'wna40to50_phi0_wave12_phi.csv', header=0)  # .transpose()
wna25to55_phi0_wave28_phi = pd.read_csv(path+'wna25to55_phi0_wave28_phi.csv', header=0)  # .transpose()
wna0to60_phi0_wave52_phi = pd.read_csv(path+'wna0to60_phi0_wave52_phi.csv', header=0)  # .transpose()


freq = np.array(list(map(float, wna45_phi0_wave4_phi.columns.tolist())))

# display(wna45_phi0_wave4_phi.describe())
names_phi = [wna45_phi0_wave4_phi, wna40to50_phi0_wave12_phi, wna25to55_phi0_wave28_phi, wna0to60_phi0_wave52_phi]


path = '/Users/ampuku/Documents/duct/code/python/for_planarity_fwhm/csv/rand_fconst_'
wna45_phi0_wave4_theta = pd.read_csv(path+'wna45_phi0_wave4_theta.csv', header=0)  # .transpose()
wna40to50_phi0_wave12_theta = pd.read_csv(path+'wna40to50_phi0_wave12_theta.csv', header=0)  # .transpose()
wna25to55_phi0_wave28_theta = pd.read_csv(path+'wna25to55_phi0_wave28_theta.csv', header=0)  # .transpose()
wna0to60_phi0_wave52_theta = pd.read_csv(path+'wna0to60_phi0_wave52_theta.csv', header=0)  # .transpose()


freq = np.array(list(map(float, wna45_phi0_wave4_theta.columns.tolist())))

# display(wna45_phi0_wave4_theta.describe())
names_theta = [wna45_phi0_wave4_theta, wna40to50_phi0_wave12_theta, wna25to55_phi0_wave28_theta, wna0to60_phi0_wave52_theta]


path = '/Users/ampuku/Documents/duct/code/python/for_planarity_fwhm/csv/rand_fconst_'
wna45_phi0_wave4_pla = pd.read_csv(path+'wna45_phi0_wave4_pla.csv', header=0)  # .transpose()
wna40to50_phi0_wave12_pla = pd.read_csv(path+'wna40to50_phi0_wave12_pla.csv', header=0)  # .transpose()
wna25to55_phi0_wave28_pla = pd.read_csv(path+'wna25to55_phi0_wave28_pla.csv', header=0)  # .transpose()
wna0to60_phi0_wave52_pla = pd.read_csv(path+'wna0to60_phi0_wave52_pla.csv', header=0)  # .transpose()


freq = np.array(list(map(float, wna45_phi0_wave4_pla.columns.tolist())))

# display(wna45_phi0_wave4_pla.describe())
names_pla = [wna45_phi0_wave4_pla, wna40to50_phi0_wave12_pla, wna25to55_phi0_wave28_pla, wna0to60_phi0_wave52_pla]


forcus_f = 1024.0
theta_input = np.arange(5., 40., 10.)
theta_input_ave = (np.array([45., 40., 25., 0.])+np.array([45., 50., 55., 60.])) / 2
theta_output_all = []
phi_output_all = []
pla_output_all = []

for i in range(len(theta_input)):
    theta_output_all.append(names_theta[i].iloc[:, 16])
    phi_output_all.append(names_phi[i].iloc[:, 16])
    pla_output_all.append(names_pla[i].iloc[:, 16])


# fig = plt.figure(figsize=(5, 15))
ax1 = fig.add_subplot(3, 3, 3)
ax1.set_title('phi : 0 deg\n', size=20)
axplots(ax1, theta_output_all, theta_input, theta_input_ave)
# ax1.set_xlabel('input theta [degree]')
# ax1.set_ylabel('mag SVD theta [degree]')
ax1.set_ylim([0, 90])
ax1.grid()
ax2 = fig.add_subplot(3, 3, 6)
axplots(ax2, phi_output_all, theta_input, theta_input*0)
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
# plt.savefig('/Users/ampuku/Documents/duct/Fig/calcs/rand_fconst_wnato_wave4n_withv')
plt.savefig('/Users/ampuku/Documents/duct/Fig/calcs/rand_fconst_wnato_wave4n_35_v2')


# %%
