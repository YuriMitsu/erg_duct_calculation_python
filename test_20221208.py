# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import japanize_matplotlib


"""
波の数=8に固定
phi=[0.,45.,90.,135.,180.,225.,270.,315.]の等方な分布
横軸:入力値のWNA,10度刻みで変化させてみる
f/fc=0.2, theta_g=66.4
縦軸:SVD結果のplanarity,WNA,phi

"""

path = '/Users/ampuku/Documents/duct/code/python/for_planarity_fwhm/csv/rand_fconst_'
wna5_phi0360_wave8_phi = pd.read_csv(path+'wna5_phi0360_wave8_phi.csv', header=0)  # .transpose()
wna15_phi0360_wave8_phi = pd.read_csv(path+'wna15_phi0360_wave8_phi.csv', header=0)  # .transpose()
wna25_phi0360_wave8_phi = pd.read_csv(path+'wna25_phi0360_wave8_phi.csv', header=0)  # .transpose()
wna35_phi0360_wave8_phi = pd.read_csv(path+'wna35_phi0360_wave8_phi.csv', header=0)  # .transpose()
wna45_phi0360_wave8_phi = pd.read_csv(path+'wna45_phi0360_wave8_phi.csv', header=0)  # .transpose()
wna55_phi0360_wave8_phi = pd.read_csv(path+'wna55_phi0360_wave8_phi.csv', header=0)  # .transpose()
wna65_phi0360_wave8_phi = pd.read_csv(path+'wna65_phi0360_wave8_phi.csv', header=0)  # .transpose()
wna75_phi0360_wave8_phi = pd.read_csv(path+'wna75_phi0360_wave8_phi.csv', header=0)  # .transpose()

freq = np.array(list(map(float, wna5_phi0360_wave8_phi.columns.tolist())))

# display(wna5_phi0360_wave8_phi.describe())
names_phi = [wna5_phi0360_wave8_phi, wna15_phi0360_wave8_phi, wna25_phi0360_wave8_phi, wna35_phi0360_wave8_phi,
             wna45_phi0360_wave8_phi, wna55_phi0360_wave8_phi, wna65_phi0360_wave8_phi, wna75_phi0360_wave8_phi]
# for name in names_phi:
    # plt.plot(freq[(freq > 200.) & (freq < 1000.)], name.mean()[(freq > 200.) & (freq < 1000.)])
    # plt.plot(freq[(freq > 200.) & (freq < 1000.)], name.std()[(freq > 200.) & (freq < 1000.)])
    # plt.show()



path = '/Users/ampuku/Documents/duct/code/python/for_planarity_fwhm/csv/rand_fconst_'
wna5_phi0360_wave8_theta = pd.read_csv(path+'wna5_phi0360_wave8_theta.csv', header=0)  # .transpose()
wna15_phi0360_wave8_theta = pd.read_csv(path+'wna15_phi0360_wave8_theta.csv', header=0)  # .transpose()
wna25_phi0360_wave8_theta = pd.read_csv(path+'wna25_phi0360_wave8_theta.csv', header=0)  # .transpose()
wna35_phi0360_wave8_theta = pd.read_csv(path+'wna35_phi0360_wave8_theta.csv', header=0)  # .transpose()
wna45_phi0360_wave8_theta = pd.read_csv(path+'wna45_phi0360_wave8_theta.csv', header=0)  # .transpose()
wna55_phi0360_wave8_theta = pd.read_csv(path+'wna55_phi0360_wave8_theta.csv', header=0)  # .transpose()
wna65_phi0360_wave8_theta = pd.read_csv(path+'wna65_phi0360_wave8_theta.csv', header=0)  # .transpose()
wna75_phi0360_wave8_theta = pd.read_csv(path+'wna75_phi0360_wave8_theta.csv', header=0)  # .transpose()

freq = np.array(list(map(float, wna5_phi0360_wave8_theta.columns.tolist())))

# display(wna5_phi0360_wave8_theta.describe())
names_theta = [wna5_phi0360_wave8_theta, wna15_phi0360_wave8_theta, wna25_phi0360_wave8_theta, wna35_phi0360_wave8_theta,
               wna45_phi0360_wave8_theta, wna55_phi0360_wave8_theta, wna65_phi0360_wave8_theta, wna75_phi0360_wave8_theta]
# for name in names_theta:
    # plt.plot(freq[(freq > 200.) & (freq < 1000.)], name.mean()[(freq > 200.) & (freq < 1000.)])
    # plt.plot(freq[(freq > 200.) & (freq < 1000.)], name.std()[(freq > 200.) & (freq < 1000.)])
    # plt.show()




path = '/Users/ampuku/Documents/duct/code/python/for_planarity_fwhm/csv/rand_fconst_'
wna5_phi0360_wave8_pla = pd.read_csv(path+'wna5_phi0360_wave8_pla.csv', header=0)  # .transpose()
wna15_phi0360_wave8_pla = pd.read_csv(path+'wna15_phi0360_wave8_pla.csv', header=0)  # .transpose()
wna25_phi0360_wave8_pla = pd.read_csv(path+'wna25_phi0360_wave8_pla.csv', header=0)  # .transpose()
wna35_phi0360_wave8_pla = pd.read_csv(path+'wna35_phi0360_wave8_pla.csv', header=0)  # .transpose()
wna45_phi0360_wave8_pla = pd.read_csv(path+'wna45_phi0360_wave8_pla.csv', header=0)  # .transpose()
wna55_phi0360_wave8_pla = pd.read_csv(path+'wna55_phi0360_wave8_pla.csv', header=0)  # .transpose()
wna65_phi0360_wave8_pla = pd.read_csv(path+'wna65_phi0360_wave8_pla.csv', header=0)  # .transpose()
wna75_phi0360_wave8_pla = pd.read_csv(path+'wna75_phi0360_wave8_pla.csv', header=0)  # .transpose()

freq = np.array(list(map(float, wna5_phi0360_wave8_pla.columns.tolist())))

# display(wna5_phi0360_wave8_pla.describe())
names_pla = [wna5_phi0360_wave8_pla, wna15_phi0360_wave8_pla, wna25_phi0360_wave8_pla, wna35_phi0360_wave8_pla,
             wna45_phi0360_wave8_pla, wna55_phi0360_wave8_pla, wna65_phi0360_wave8_pla, wna75_phi0360_wave8_pla]
# for name in names_pla:
    # plt.plot(freq[(freq > 200.) & (freq < 1000.)], name.mean()[(freq > 200.) & (freq < 1000.)])
    # plt.plot(freq[(freq > 200.) & (freq < 1000.)], name.std()[(freq > 200.) & (freq < 1000.)])
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


fig = plt.figure(figsize=(5, 5))
ax3 = fig.add_subplot(1, 1, 1)
# ax3.set_title('num of waves : 8, phi : 0-360 deg\nfft区間ごとの位相がランダム、周波数一定', size=15)
ax3.errorbar(theta_input, theta_output_mean, yerr=theta_output_std, capsize=3, fmt='o', ecolor='k', ms=7, mfc='None', mec='k')
ax3.plot(theta_input, theta_input, color='gray')
ax3.set_xlabel('input WNA [degree]', size=22)
ax3.set_ylabel('SVD analysis WNA [degree]', size=22)
ax3.set_xlim([0, 80])
ax3.set_xticklabels([0,10,20,30,40,50,60,70,80], size=16)
ax3.set_ylim([0, 90])
ax3.set_yticklabels([0,10,20,30,40,50,60,70,80, 90], size=16)
ax3.grid()

plt.tight_layout()
# plt.savefig('/Users/ampuku/Documents/duct/Fig/calcs/rand_fconst_wna575_phi0360_wave8')



# %%
fig = plt.figure(figsize=(5, 5))

ax3 = fig.add_subplot(1, 1, 1)
ax3.errorbar(theta_input, pla_output_mean, yerr=pla_output_std, capsize=3, fmt='o', ecolor='k', ms=7, mfc='None', mec='k')
ax3.set_xlabel('input WNA [degree]', size=22)
ax3.set_ylabel('SVD analysis planarity [degree]', size=22)
ax3.set_xlim([0, 80])
ax3.set_xticklabels([0,10,20,30,40,50,60,70,80], size=16)
ax3.set_yticks([0.2, 0.4, 0.6, 0.8, 1])
ax3.set_yticklabels([0.2, 0.4, 0.6, 0.8, 1.0], size=16)
# ax3.set_ylim([0.2, 1])
# ax3.set_yticklabels([0,10,20,30,40,50,60,70,80, 90], size=16)
ax3.grid()

plt.tight_layout()
# plt.savefig('/Users/ampuku/Documents/duct/Fig/calcs/rand_fconst_wna575_phi0360_wave8')






# %%
