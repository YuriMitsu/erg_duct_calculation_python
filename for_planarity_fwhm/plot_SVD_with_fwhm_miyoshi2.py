# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



# %%

"""
波の数=8に固定
phi=[0.,0.,0.,0.,0.,0.,0.,0.]の等方な分布
thetaを180 degずらして4波ずつ
横軸:入力値のWNA,10度刻みで変化させてみる
f/fc=0.2, theta_g=66.4
縦軸:SVD結果のplanarity,WNA,phi

"""

path = '/Users/ampuku/Documents/duct/code/python/for_planarity_fwhm/csv/'
wna0180_phi0_wave8_phi = pd.read_csv(path+'rand_fconst_wna0180_phi0_wave8_phi.csv', header=0)  # .transpose()
wna5185_phi0_wave8_phi = pd.read_csv(path+'rand_fconst_wna5185_phi0_wave8_phi.csv', header=0)  # .transpose()
wna15195_phi0_wave8_phi = pd.read_csv(path+'rand_fconst_wna15195_phi0_wave8_phi.csv', header=0)  # .transpose()
wna25205_phi0_wave8_phi = pd.read_csv(path+'rand_fconst_wna25205_phi0_wave8_phi.csv', header=0)  # .transpose()
wna35215_phi0_wave8_phi = pd.read_csv(path+'rand_fconst_wna35215_phi0_wave8_phi.csv', header=0)  # .transpose()
wna45225_phi0_wave8_phi = pd.read_csv(path+'rand_fconst_wna45225_phi0_wave8_phi.csv', header=0)  # .transpose()
wna55235_phi0_wave8_phi = pd.read_csv(path+'rand_fconst_wna55235_phi0_wave8_phi.csv', header=0)  # .transpose()
wna65245_phi0_wave8_phi = pd.read_csv(path+'rand_fconst_wna65245_phi0_wave8_phi.csv', header=0)  # .transpose()
wna75255_phi0_wave8_phi = pd.read_csv(path+'rand_fconst_wna75255_phi0_wave8_phi.csv', header=0)  # .transpose()

freq = np.array(list(map(float, wna5185_phi0_wave8_phi.columns.tolist())))

# display(wna5_phi0_wave8_phi.describe())
names_phi = [wna0180_phi0_wave8_phi, wna5185_phi0_wave8_phi, wna15195_phi0_wave8_phi, wna25205_phi0_wave8_phi, wna35215_phi0_wave8_phi, wna45225_phi0_wave8_phi, wna55235_phi0_wave8_phi, wna65245_phi0_wave8_phi, wna75255_phi0_wave8_phi]
# for name in names_phi:
# plt.plot(freq[(freq>200.) & (freq<1000.)], name.mean()[(freq>200.) & (freq<1000.)])
# plt.plot(freq[(freq>200.) & (freq<1000.)], name.std()[(freq>200.) & (freq<1000.)])
# plt.show()



path = '/Users/ampuku/Documents/duct/code/python/for_planarity_fwhm/csv/'
wna5_phi0_wave8_theta = pd.read_csv(path+'rand_fconst_wna0180_phi0_wave8_theta.csv', header=0)  # .transpose()
wna5_phi0_wave8_theta = pd.read_csv(path+'rand_fconst_wna5185_phi0_wave8_theta.csv', header=0)  # .transpose()
wna15_phi0_wave8_theta = pd.read_csv(path+'rand_fconst_wna15195_phi0_wave8_theta.csv', header=0)  # .transpose()
wna25_phi0_wave8_theta = pd.read_csv(path+'rand_fconst_wna25205_phi0_wave8_theta.csv', header=0)  # .transpose()
wna35_phi0_wave8_theta = pd.read_csv(path+'rand_fconst_wna35215_phi0_wave8_theta.csv', header=0)  # .transpose()
wna45_phi0_wave8_theta = pd.read_csv(path+'rand_fconst_wna45225_phi0_wave8_theta.csv', header=0)  # .transpose()
wna55_phi0_wave8_theta = pd.read_csv(path+'rand_fconst_wna55235_phi0_wave8_theta.csv', header=0)  # .transpose()
wna65_phi0_wave8_theta = pd.read_csv(path+'rand_fconst_wna65245_phi0_wave8_theta.csv', header=0)  # .transpose()
wna75_phi0_wave8_theta = pd.read_csv(path+'rand_fconst_wna75255_phi0_wave8_theta.csv', header=0)  # .transpose()

freq = np.array(list(map(float, wna5_phi0_wave8_theta.columns.tolist())))

# display(wna5_phi0_wave8_theta.describe())
names_theta = [wna5_phi0_wave8_theta, wna15_phi0_wave8_theta, wna25_phi0_wave8_theta, wna35_phi0_wave8_theta, wna45_phi0_wave8_theta, wna55_phi0_wave8_theta, wna65_phi0_wave8_theta, wna75_phi0_wave8_theta]
# for name in names_theta:
# plt.plot(freq[(freq>200.) & (freq<1000.)], name.mean()[(freq>200.) & (freq<1000.)])
# plt.plot(freq[(freq>200.) & (freq<1000.)], name.std()[(freq>200.) & (freq<1000.)])
# plt.show()




path = '/Users/ampuku/Documents/duct/code/python/for_planarity_fwhm/csv/'
wna5_phi0_wave8_pla = pd.read_csv(path+'rand_fconst_wna0180_phi0_wave8_pla.csv', header=0)  # .transpose()
wna5_phi0_wave8_pla = pd.read_csv(path+'rand_fconst_wna5185_phi0_wave8_pla.csv', header=0)  # .transpose()
wna15_phi0_wave8_pla = pd.read_csv(path+'rand_fconst_wna15195_phi0_wave8_pla.csv', header=0)  # .transpose()
wna25_phi0_wave8_pla = pd.read_csv(path+'rand_fconst_wna25205_phi0_wave8_pla.csv', header=0)  # .transpose()
wna35_phi0_wave8_pla = pd.read_csv(path+'rand_fconst_wna35215_phi0_wave8_pla.csv', header=0)  # .transpose()
wna45_phi0_wave8_pla = pd.read_csv(path+'rand_fconst_wna45225_phi0_wave8_pla.csv', header=0)  # .transpose()
wna55_phi0_wave8_pla = pd.read_csv(path+'rand_fconst_wna55235_phi0_wave8_pla.csv', header=0)  # .transpose()
wna65_phi0_wave8_pla = pd.read_csv(path+'rand_fconst_wna65245_phi0_wave8_pla.csv', header=0)  # .transpose()
wna75_phi0_wave8_pla = pd.read_csv(path+'rand_fconst_wna75255_phi0_wave8_pla.csv', header=0)  # .transpose()

freq = np.array(list(map(float, wna5_phi0_wave8_pla.columns.tolist())))

# display(wna5_phi0_wave8_pla.describe())
names_pla = [wna5_phi0_wave8_pla, wna15_phi0_wave8_pla, wna25_phi0_wave8_pla, wna35_phi0_wave8_pla, wna45_phi0_wave8_pla, wna55_phi0_wave8_pla, wna65_phi0_wave8_pla, wna75_phi0_wave8_pla]
# for name in names_pla:
# plt.plot(freq[(freq>200.) & (freq<1000.)], name.mean()[(freq>200.) & (freq<1000.)])
# plt.plot(freq[(freq>200.) & (freq<1000.)], name.std()[(freq>200.) & (freq<1000.)])
# plt.show()



forcus_f = 576.0
theta_input = np.arange(5., 80., 10.)
theta_output_mean = []
theta_output_std = []
phi_output_mean = []
phi_output_std = []
pla_output_mean = []
pla_output_std = []
for i in range(len(theta_input)):
    theta_output_mean.append(names_theta[i].mean()[freq == forcus_f][0])
    theta_output_std.append(names_theta[i].std()[freq == forcus_f][0])
    phi_output_mean.append(names_phi[i].mean()[freq == forcus_f][0])
    phi_output_std.append(names_phi[i].std()[freq == forcus_f][0])
    pla_output_mean.append(names_pla[i].mean()[freq == forcus_f][0])
    pla_output_std.append(names_pla[i].std()[freq == forcus_f][0])


fig = plt.figure(figsize=(15,15))
ax1 = fig.add_subplot(3, 3, 3)
ax1.set_title('num of waves : 8, phi : 0 deg\nantiparallel\nfft区間ごとの位相がランダム、周波数一定', size=15)
ax1.errorbar(theta_input, theta_output_mean, yerr=theta_output_std, capsize=3, fmt='o', ecolor='k', ms=7, mfc='None', mec='k')
ax1.plot(theta_input, theta_input, color='gray')
ax1.set_xlabel('input theta [degree]')
ax1.set_ylabel('mag SVD theta [degree]')
ax1.set_xlim([0, 80])
ax1.set_ylim([0, 90])
ax1.grid()
ax2 = fig.add_subplot(3, 3, 6)
ax2.errorbar(theta_input, phi_output_mean, yerr=phi_output_std, capsize=3, fmt='o', ecolor='k', ms=7, mfc='None', mec='k')
ax2.plot(theta_input, theta_input*0, color='gray')
ax2.set_xlabel('input theta [degree]')
ax2.set_ylabel('mag SVD phi [degree]')
ax2.set_xlim([0, 80])
ax2.set_ylim([-180, 180])
ax2.grid()
ax3 = fig.add_subplot(3, 3, 9)
ax3.errorbar(theta_input, pla_output_mean, yerr=pla_output_std, capsize=3, fmt='o', ecolor='k', ms=7, mfc='None', mec='k')
ax3.set_xlabel('input theta [degree]')
ax3.set_ylabel('mag SVD planarity')
ax3.set_xlim([0, 80])
ax3.set_ylim([0.3, 1.0])
ax3.grid()

# plt.tight_layout()
# plt.savefig('/Users/ampuku/Documents/duct/Fig/calcs/wna0180_phi0_wave8')
# plt.show()



"""
波の数=8に固定
phi=[0.,0.,0.,0.,0.,0.,0.,0.]の等方な分布
thetaを180 degずらして4波ずつ
横軸:入力値のWNA,10度刻みで変化させてみる
f/fc=0.2, theta_g=66.4
縦軸:SVD結果のplanarity,WNA,phi

"""

path = '/Users/ampuku/Documents/duct/code/python/for_planarity_fwhm/csv/'
wna0180_phi030_wave8_phi = pd.read_csv(path+'rand_fconst_wna0180_phi030_wave8_phi.csv', header=0)  # .transpose()
wna5185_phi030_wave8_phi = pd.read_csv(path+'rand_fconst_wna5185_phi030_wave8_phi.csv', header=0)  # .transpose()
wna15195_phi030_wave8_phi = pd.read_csv(path+'rand_fconst_wna15195_phi030_wave8_phi.csv', header=0)  # .transpose()
wna25205_phi030_wave8_phi = pd.read_csv(path+'rand_fconst_wna25205_phi030_wave8_phi.csv', header=0)  # .transpose()
wna35215_phi030_wave8_phi = pd.read_csv(path+'rand_fconst_wna35215_phi030_wave8_phi.csv', header=0)  # .transpose()
wna45225_phi030_wave8_phi = pd.read_csv(path+'rand_fconst_wna45225_phi030_wave8_phi.csv', header=0)  # .transpose()
wna55235_phi030_wave8_phi = pd.read_csv(path+'rand_fconst_wna55235_phi030_wave8_phi.csv', header=0)  # .transpose()
wna65245_phi030_wave8_phi = pd.read_csv(path+'rand_fconst_wna65245_phi030_wave8_phi.csv', header=0)  # .transpose()
wna75255_phi030_wave8_phi = pd.read_csv(path+'rand_fconst_wna75255_phi030_wave8_phi.csv', header=0)  # .transpose()

freq = np.array(list(map(float, wna5185_phi030_wave8_phi.columns.tolist())))

# display(wna5_phi030_wave8_phi.describe())
names_phi = [wna0180_phi030_wave8_phi, wna5185_phi030_wave8_phi, wna15195_phi030_wave8_phi, wna25205_phi030_wave8_phi, wna35215_phi030_wave8_phi, wna45225_phi030_wave8_phi, wna55235_phi030_wave8_phi, wna65245_phi030_wave8_phi, wna75255_phi030_wave8_phi]
# for name in names_phi:
# plt.plot(freq[(freq>200.) & (freq<1000.)], name.mean()[(freq>200.) & (freq<1000.)])
# plt.plot(freq[(freq>200.) & (freq<1000.)], name.std()[(freq>200.) & (freq<1000.)])
# plt.show()



path = '/Users/ampuku/Documents/duct/code/python/for_planarity_fwhm/csv/'
wna5_phi030_wave8_theta = pd.read_csv(path+'rand_fconst_wna0180_phi030_wave8_theta.csv', header=0)  # .transpose()
wna5_phi030_wave8_theta = pd.read_csv(path+'rand_fconst_wna5185_phi030_wave8_theta.csv', header=0)  # .transpose()
wna15_phi030_wave8_theta = pd.read_csv(path+'rand_fconst_wna15195_phi030_wave8_theta.csv', header=0)  # .transpose()
wna25_phi030_wave8_theta = pd.read_csv(path+'rand_fconst_wna25205_phi030_wave8_theta.csv', header=0)  # .transpose()
wna35_phi030_wave8_theta = pd.read_csv(path+'rand_fconst_wna35215_phi030_wave8_theta.csv', header=0)  # .transpose()
wna45_phi030_wave8_theta = pd.read_csv(path+'rand_fconst_wna45225_phi030_wave8_theta.csv', header=0)  # .transpose()
wna55_phi030_wave8_theta = pd.read_csv(path+'rand_fconst_wna55235_phi030_wave8_theta.csv', header=0)  # .transpose()
wna65_phi030_wave8_theta = pd.read_csv(path+'rand_fconst_wna65245_phi030_wave8_theta.csv', header=0)  # .transpose()
wna75_phi030_wave8_theta = pd.read_csv(path+'rand_fconst_wna75255_phi030_wave8_theta.csv', header=0)  # .transpose()

freq = np.array(list(map(float, wna5_phi030_wave8_theta.columns.tolist())))

# display(wna5_phi030_wave8_theta.describe())
names_theta = [wna5_phi030_wave8_theta, wna15_phi030_wave8_theta, wna25_phi030_wave8_theta, wna35_phi030_wave8_theta, wna45_phi030_wave8_theta, wna55_phi030_wave8_theta, wna65_phi030_wave8_theta, wna75_phi030_wave8_theta]
# for name in names_theta:
# plt.plot(freq[(freq>200.) & (freq<1000.)], name.mean()[(freq>200.) & (freq<1000.)])
# plt.plot(freq[(freq>200.) & (freq<1000.)], name.std()[(freq>200.) & (freq<1000.)])
# plt.show()




path = '/Users/ampuku/Documents/duct/code/python/for_planarity_fwhm/csv/'
wna5_phi030_wave8_pla = pd.read_csv(path+'rand_fconst_wna0180_phi030_wave8_pla.csv', header=0)  # .transpose()
wna5_phi030_wave8_pla = pd.read_csv(path+'rand_fconst_wna5185_phi030_wave8_pla.csv', header=0)  # .transpose()
wna15_phi030_wave8_pla = pd.read_csv(path+'rand_fconst_wna15195_phi030_wave8_pla.csv', header=0)  # .transpose()
wna25_phi030_wave8_pla = pd.read_csv(path+'rand_fconst_wna25205_phi030_wave8_pla.csv', header=0)  # .transpose()
wna35_phi030_wave8_pla = pd.read_csv(path+'rand_fconst_wna35215_phi030_wave8_pla.csv', header=0)  # .transpose()
wna45_phi030_wave8_pla = pd.read_csv(path+'rand_fconst_wna45225_phi030_wave8_pla.csv', header=0)  # .transpose()
wna55_phi030_wave8_pla = pd.read_csv(path+'rand_fconst_wna55235_phi030_wave8_pla.csv', header=0)  # .transpose()
wna65_phi030_wave8_pla = pd.read_csv(path+'rand_fconst_wna65245_phi030_wave8_pla.csv', header=0)  # .transpose()
wna75_phi030_wave8_pla = pd.read_csv(path+'rand_fconst_wna75255_phi030_wave8_pla.csv', header=0)  # .transpose()

freq = np.array(list(map(float, wna5_phi030_wave8_pla.columns.tolist())))

# display(wna5_phi030_wave8_pla.describe())
names_pla = [wna5_phi030_wave8_pla, wna15_phi030_wave8_pla, wna25_phi030_wave8_pla, wna35_phi030_wave8_pla, wna45_phi030_wave8_pla, wna55_phi030_wave8_pla, wna65_phi030_wave8_pla, wna75_phi030_wave8_pla]
# for name in names_pla:
# plt.plot(freq[(freq>200.) & (freq<1000.)], name.mean()[(freq>200.) & (freq<1000.)])
# plt.plot(freq[(freq>200.) & (freq<1000.)], name.std()[(freq>200.) & (freq<1000.)])
# plt.show()



forcus_f = 576.0
theta_input = np.arange(5., 80., 10.)
theta_output_mean = []
theta_output_std = []
phi_output_mean = []
phi_output_std = []
pla_output_mean = []
pla_output_std = []
for i in range(len(theta_input)):
    theta_output_mean.append(names_theta[i].mean()[freq == forcus_f][0])
    theta_output_std.append(names_theta[i].std()[freq == forcus_f][0])
    phi_output_mean.append(names_phi[i].mean()[freq == forcus_f][0])
    phi_output_std.append(names_phi[i].std()[freq == forcus_f][0])
    pla_output_mean.append(names_pla[i].mean()[freq == forcus_f][0])
    pla_output_std.append(names_pla[i].std()[freq == forcus_f][0])


# fig = plt.figure(figsize=(5,15))
ax1 = fig.add_subplot(3, 3, 2)
ax1.set_title('num of waves : 8, phi : 0-30 deg\nantiparallel\nfft区間ごとの位相がランダム、周波数一定', size=15)
ax1.errorbar(theta_input, theta_output_mean, yerr=theta_output_std, capsize=3, fmt='o', ecolor='k', ms=7, mfc='None', mec='k')
ax1.plot(theta_input, theta_input, color='gray')
ax1.set_xlabel('input theta [degree]')
ax1.set_ylabel('mag SVD theta [degree]')
ax1.set_xlim([0, 80])
ax1.set_ylim([0, 90])
ax1.grid()
ax2 = fig.add_subplot(3, 3, 5)
ax2.errorbar(theta_input, phi_output_mean, yerr=phi_output_std, capsize=3, fmt='o', ecolor='k', ms=7, mfc='None', mec='k')
ax2.plot(theta_input, theta_input*0, color='gray')
ax2.set_xlabel('input theta [degree]')
ax2.set_ylabel('mag SVD phi [degree]')
ax2.set_xlim([0, 80])
ax2.set_ylim([-180, 180])
ax2.grid()
ax3 = fig.add_subplot(3, 3, 8)
ax3.errorbar(theta_input, pla_output_mean, yerr=pla_output_std, capsize=3, fmt='o', ecolor='k', ms=7, mfc='None', mec='k')
ax3.set_xlabel('input theta [degree]')
ax3.set_ylabel('mag SVD planarity')
ax3.set_xlim([0, 80])
ax3.set_ylim([0.3, 1.0])
ax3.grid()

# plt.tight_layout()
# plt.savefig('/Users/ampuku/Documents/duct/Fig/calcs/wna0180_phi030_wave8')
# plt.show()






"""
波の数=8に固定
phi=[0.,0.,0.,0.,0.,0.,0.,0.]の等方な分布
thetaを180 degずらして4波ずつ
横軸:入力値のWNA,10度刻みで変化させてみる
f/fc=0.2, theta_g=66.4
縦軸:SVD結果のplanarity,WNA,phi

"""

path = '/Users/ampuku/Documents/duct/code/python/for_planarity_fwhm/csv/'
wna0180_phi0360_wave8_phi = pd.read_csv(path+'rand_fconst_wna0180_phi0360_wave8_phi.csv', header=0)  # .transpose()
wna5185_phi0360_wave8_phi = pd.read_csv(path+'rand_fconst_wna5185_phi0360_wave8_phi.csv', header=0)  # .transpose()
wna15195_phi0360_wave8_phi = pd.read_csv(path+'rand_fconst_wna15195_phi0360_wave8_phi.csv', header=0)  # .transpose()
wna25205_phi0360_wave8_phi = pd.read_csv(path+'rand_fconst_wna25205_phi0360_wave8_phi.csv', header=0)  # .transpose()
wna35215_phi0360_wave8_phi = pd.read_csv(path+'rand_fconst_wna35215_phi0360_wave8_phi.csv', header=0)  # .transpose()
wna45225_phi0360_wave8_phi = pd.read_csv(path+'rand_fconst_wna45225_phi0360_wave8_phi.csv', header=0)  # .transpose()
wna55235_phi0360_wave8_phi = pd.read_csv(path+'rand_fconst_wna55235_phi0360_wave8_phi.csv', header=0)  # .transpose()
wna65245_phi0360_wave8_phi = pd.read_csv(path+'rand_fconst_wna65245_phi0360_wave8_phi.csv', header=0)  # .transpose()
wna75255_phi0360_wave8_phi = pd.read_csv(path+'rand_fconst_wna75255_phi0360_wave8_phi.csv', header=0)  # .transpose()

freq = np.array(list(map(float, wna5185_phi0360_wave8_phi.columns.tolist())))

# display(wna5_phi0360_wave8_phi.describe())
names_phi = [wna0180_phi0360_wave8_phi, wna5185_phi0360_wave8_phi, wna15195_phi0360_wave8_phi, wna25205_phi0360_wave8_phi, wna35215_phi0360_wave8_phi, wna45225_phi0360_wave8_phi, wna55235_phi0360_wave8_phi, wna65245_phi0360_wave8_phi, wna75255_phi0360_wave8_phi]
# for name in names_phi:
# plt.plot(freq[(freq>200.) & (freq<1000.)], name.mean()[(freq>200.) & (freq<1000.)])
# plt.plot(freq[(freq>200.) & (freq<1000.)], name.std()[(freq>200.) & (freq<1000.)])
# plt.show()



path = '/Users/ampuku/Documents/duct/code/python/for_planarity_fwhm/csv/'
wna5_phi0360_wave8_theta = pd.read_csv(path+'rand_fconst_wna0180_phi0360_wave8_theta.csv', header=0)  # .transpose()
wna5_phi0360_wave8_theta = pd.read_csv(path+'rand_fconst_wna5185_phi0360_wave8_theta.csv', header=0)  # .transpose()
wna15_phi0360_wave8_theta = pd.read_csv(path+'rand_fconst_wna15195_phi0360_wave8_theta.csv', header=0)  # .transpose()
wna25_phi0360_wave8_theta = pd.read_csv(path+'rand_fconst_wna25205_phi0360_wave8_theta.csv', header=0)  # .transpose()
wna35_phi0360_wave8_theta = pd.read_csv(path+'rand_fconst_wna35215_phi0360_wave8_theta.csv', header=0)  # .transpose()
wna45_phi0360_wave8_theta = pd.read_csv(path+'rand_fconst_wna45225_phi0360_wave8_theta.csv', header=0)  # .transpose()
wna55_phi0360_wave8_theta = pd.read_csv(path+'rand_fconst_wna55235_phi0360_wave8_theta.csv', header=0)  # .transpose()
wna65_phi0360_wave8_theta = pd.read_csv(path+'rand_fconst_wna65245_phi0360_wave8_theta.csv', header=0)  # .transpose()
wna75_phi0360_wave8_theta = pd.read_csv(path+'rand_fconst_wna75255_phi0360_wave8_theta.csv', header=0)  # .transpose()

freq = np.array(list(map(float, wna5_phi0360_wave8_theta.columns.tolist())))

# display(wna5_phi0360_wave8_theta.describe())
names_theta = [wna5_phi0360_wave8_theta, wna15_phi0360_wave8_theta, wna25_phi0360_wave8_theta, wna35_phi0360_wave8_theta, wna45_phi0360_wave8_theta, wna55_phi0360_wave8_theta, wna65_phi0360_wave8_theta, wna75_phi0360_wave8_theta]
# for name in names_theta:
# plt.plot(freq[(freq>200.) & (freq<1000.)], name.mean()[(freq>200.) & (freq<1000.)])
# plt.plot(freq[(freq>200.) & (freq<1000.)], name.std()[(freq>200.) & (freq<1000.)])
# plt.show()




path = '/Users/ampuku/Documents/duct/code/python/for_planarity_fwhm/csv/'
wna5_phi0360_wave8_pla = pd.read_csv(path+'rand_fconst_wna0180_phi0360_wave8_pla.csv', header=0)  # .transpose()
wna5_phi0360_wave8_pla = pd.read_csv(path+'rand_fconst_wna5185_phi0360_wave8_pla.csv', header=0)  # .transpose()
wna15_phi0360_wave8_pla = pd.read_csv(path+'rand_fconst_wna15195_phi0360_wave8_pla.csv', header=0)  # .transpose()
wna25_phi0360_wave8_pla = pd.read_csv(path+'rand_fconst_wna25205_phi0360_wave8_pla.csv', header=0)  # .transpose()
wna35_phi0360_wave8_pla = pd.read_csv(path+'rand_fconst_wna35215_phi0360_wave8_pla.csv', header=0)  # .transpose()
wna45_phi0360_wave8_pla = pd.read_csv(path+'rand_fconst_wna45225_phi0360_wave8_pla.csv', header=0)  # .transpose()
wna55_phi0360_wave8_pla = pd.read_csv(path+'rand_fconst_wna55235_phi0360_wave8_pla.csv', header=0)  # .transpose()
wna65_phi0360_wave8_pla = pd.read_csv(path+'rand_fconst_wna65245_phi0360_wave8_pla.csv', header=0)  # .transpose()
wna75_phi0360_wave8_pla = pd.read_csv(path+'rand_fconst_wna75255_phi0360_wave8_pla.csv', header=0)  # .transpose()

freq = np.array(list(map(float, wna5_phi0360_wave8_pla.columns.tolist())))

# display(wna5_phi0360_wave8_pla.describe())
names_pla = [wna5_phi0360_wave8_pla, wna15_phi0360_wave8_pla, wna25_phi0360_wave8_pla, wna35_phi0360_wave8_pla, wna45_phi0360_wave8_pla, wna55_phi0360_wave8_pla, wna65_phi0360_wave8_pla, wna75_phi0360_wave8_pla]
# for name in names_pla:
# plt.plot(freq[(freq>200.) & (freq<1000.)], name.mean()[(freq>200.) & (freq<1000.)])
# plt.plot(freq[(freq>200.) & (freq<1000.)], name.std()[(freq>200.) & (freq<1000.)])
# plt.show()



forcus_f = 576.0
theta_input = np.arange(5., 80., 10.)
theta_output_mean = []
theta_output_std = []
phi_output_mean = []
phi_output_std = []
pla_output_mean = []
pla_output_std = []
for i in range(len(theta_input)):
    theta_output_mean.append(names_theta[i].mean()[freq == forcus_f][0])
    theta_output_std.append(names_theta[i].std()[freq == forcus_f][0])
    phi_output_mean.append(names_phi[i].mean()[freq == forcus_f][0])
    phi_output_std.append(names_phi[i].std()[freq == forcus_f][0])
    pla_output_mean.append(names_pla[i].mean()[freq == forcus_f][0])
    pla_output_std.append(names_pla[i].std()[freq == forcus_f][0])


# fig = plt.figure(figsize=(5,15))
ax1 = fig.add_subplot(3, 3, 1)
ax1.set_title('num of waves : 8, phi : 0-360 deg\nantiparallel\nfft区間ごとの位相がランダム、周波数一定', size=15)
ax1.errorbar(theta_input, theta_output_mean, yerr=theta_output_std, capsize=3, fmt='o', ecolor='k', ms=7, mfc='None', mec='k')
ax1.plot(theta_input, theta_input, color='gray')
ax1.set_xlabel('input theta [degree]')
ax1.set_ylabel('mag SVD theta [degree]')
ax1.set_xlim([0, 80])
ax1.set_ylim([0, 90])
ax1.grid()
ax2 = fig.add_subplot(3, 3, 4)
ax2.errorbar(theta_input, phi_output_mean, yerr=phi_output_std, capsize=3, fmt='o', ecolor='k', ms=7, mfc='None', mec='k')
ax2.plot(theta_input, theta_input*0, color='gray')
ax2.set_xlabel('input theta [degree]')
ax2.set_ylabel('mag SVD phi [degree]')
ax2.set_xlim([0, 80])
ax2.set_ylim([-180, 180])
ax2.grid()
ax3 = fig.add_subplot(3, 3, 7)
ax3.errorbar(theta_input, pla_output_mean, yerr=pla_output_std, capsize=3, fmt='o', ecolor='k', ms=7, mfc='None', mec='k')
ax3.set_xlabel('input theta [degree]')
ax3.set_ylabel('mag SVD planarity')
ax3.set_xlim([0, 80])
ax3.set_ylim([0.3, 1.0])
ax3.grid()

plt.tight_layout()
# plt.savefig('/Users/ampuku/Documents/duct/Fig/calcs/antiparallel_wna0180_phi0360_wave8')
plt.savefig('/Users/ampuku/Documents/duct/Fig/calcs/antiparallel_rand_fconst_wna0180_wave8')
plt.show()

# %%
