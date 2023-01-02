# %%
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
%matplotlib tk

# %%
'''
電子密度データを保存して読み込み

idl
erg_init
timespan, '2018-06-06/11:25:00', 20, /min
calc_Ne, UHR_file_name='kuma'
calc_fce_and_flhr
calc_wave_params
calc_kpara
tplot, ['Ne', 'erg_mgf_l2_magt_8sec']
get_data, 'Ne', data=ndata
get_data, 'erg_mgf_l2_magt_8sec', data=bdata

stt = time_double('2018-06-06/11:25:00')
ett = time_double('2018-06-06/11:45:00')
ntime_res =  ndata.x[100]- ndata.x[99]
idx_t = where( ndata.x lt stt+ntime_res and ndata.x gt stt-ntime_res, cnt )
int(idx_t+20*60/ntime_res)
nedata = ndata.y[5135:5285]
tdata = ndata.x[5135:5285]

btime_res =  bdata.x[100]- bdata.x[99]
idx_t = where( bdata.x lt stt+btime_res and bdata.x gt stt-btime_res, cnt )
idx_t+20*60/btime_res
bbdata = bdata.y[5135:5285]
btdata = bdata.x[5135:5285]

File_path = '/Users/ampuku/Documents/duct/code/python/test_20221116/'
WRITE_CSV, File_path+'Nedata.csv', nedata
WRITE_CSV, File_path+'tdata.csv', tdata
WRITE_CSV, File_path+'bdata.csv', bbdata

'''
# %%
# 増加ダクトのlsm(kpara)の場合
path = '/Users/ampuku/Documents/duct/code/python/test_20221116/'
lsm = [0.00037135126, 0.00056528556]

Bdata = pd.read_csv(path+'bdata.csv', header=None)
B = Bdata[0]
Nedata = pd.read_csv(path+'Nedata.csv', header=None)
Ne = Nedata[0]
tdata = pd.read_csv(path+'tdata.csv', header=None)
tt = tdata[0]
farr = np.arange(1., 8., 0.1)
fce = 1 / 2 / np.pi * (1.6 * 10**(-19.)) / (9.1093 * 10**(-31.)) * B * 1e-9 / 1000

# %%
kpara_linear = lsm[0] * farr + lsm[1]

b1 = (9.1093 * 10**(-31.)) / (1.25 * 10**(-6.)) / (1.6 * 10**(-19.))**2
Ne_0 = []
Ne_1 = []
fff = []
ttt = []
nnn = []
for i in range(len(kpara_linear)):
    Ne_0.append(b1 * kpara_linear[i]**2 * (fce / farr[i] - 1) / 10**(6.))
    Ne_1.append(b1 * kpara_linear[i]**2 * (fce / (2*farr[i]))**2 / 10**(6.))
    fff.append([farr[i]]*len(fce))
    ttt.append(tt)
    nnn.append(Ne)

Ne_0 = np.array(Ne_0)
Ne_1 = np.array(Ne_1)
fff = np.array(fff)
ttt = np.array(ttt)
nnn = np.array(nnn)

# %%

# Figureと3DAxeS
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")

# 軸ラベルを設定
ax.set_xlabel("time[s]", size=16)
ax.set_ylabel("frequency[kHz]", size=16)
ax.set_zlabel("Ne[/cc]", size=16)

ax.plot_wireframe(ttt-tt[0], fff, Ne_0, color="darkblue", label='Ne0')
ax.plot_wireframe(ttt-tt[0], fff, Ne_1, color="red", label='Ne1')
ax.plot_wireframe(ttt-tt[0], fff, nnn, color="y", label='Ne')

ax.set_xlim(0., np.array(tt)[-1]-np.array(tt)[0])
ax.set_zlim(100., 400.)


plt.legend()


# %%
# 減少ダクトのlsm(kpara)の場合
lsm = [0.0003555313670662758, 0.00036820383445613147]

# %%
kpara_linear_d = lsm[0] * farr + lsm[1]

b1 = (9.1093 * 10**(-31.)) / (1.25 * 10**(-6.)) / (1.6 * 10**(-19.))**2
Ne_0d = []
Ne_1d = []
fff = []
ttt = []
for i in range(len(kpara_linear)):
    Ne_0d.append(b1 * kpara_linear_d[i]**2 * (fce / farr[i] - 1) / 10**(6.))
    Ne_1d.append(b1 * kpara_linear_d[i]**2 * (fce / (2*farr[i]))**2 / 10**(6.))
    fff.append([farr[i]]*len(fce))
    ttt.append(tt)

Ne_0d = np.array(Ne_0d)
Ne_1d = np.array(Ne_1d)
fff = np.array(fff)
ttt = np.array(ttt)

# %%

# Figureと3DAxeS
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")

# 軸ラベルを設定
ax.set_xlabel("time[s]", size=16)
ax.set_ylabel("frequency[kHz]", size=16)
ax.set_zlabel("Ne[/cc]", size=16)

ax.plot_wireframe(ttt-tt[0], fff, Ne_0d, color="darkblue", label='Ne0')
# ax.plot_wireframe(ttt-tt[0],fff,Ne_1d,color="red",label='Ne1')
ax.plot_wireframe(ttt-tt[0], fff, nnn, color="y", label='Ne')

ax.set_xlim(0., np.array(tt)[-1]-np.array(tt)[0])
ax.set_zlim(100., 400.)


plt.legend()
plt.show()

# %%
