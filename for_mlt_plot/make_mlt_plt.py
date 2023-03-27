# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pyspedas as pys
import pytplot as pyt
import japanize_matplotlib

# %%
l_val = []
gsmx = []
gsmy = []
gsmz = []
mlat = []
mlt = []

# %%
# for yyyymm in ['201704', '201705']:
for yyyymm in ['201704']:
    eventlist = pd.read_csv('/Users/ampuku/Documents/duct/code/python/event_lists/suspicion/sus_'+yyyymm+'.csv')

    for i in range(len(eventlist['start_time'])):
        sst = eventlist['start_time'][i].replace('/', ' ')
        eet = pys.time_string(pys.time_double(sst)+float(eventlist['range_min'][i])*60)
        trange = [sst, eet]
        pys.erg.orb(trange=trange)
        lmdata = pyt.get_data('erg_orb_l2_pos_Lm')
        gsmdata = pyt.get_data('erg_orb_l2_pos_gsm')

        # print(data.times[(data.times > pys.time_double(sst)) & (data.times < pys.time_double(eet))])
        l_val.extend(lmdata.y[:, 0][(lmdata.times > pys.time_double(sst)) & (lmdata.times < pys.time_double(eet))])
        gsmx.extend(gsmdata.y[:, 0][(gsmdata.times > pys.time_double(sst)) & (gsmdata.times < pys.time_double(eet))])
        gsmy.extend(gsmdata.y[:, 1][(gsmdata.times > pys.time_double(sst)) & (gsmdata.times < pys.time_double(eet))])
        gsmz.extend(gsmdata.y[:, 2][(gsmdata.times > pys.time_double(sst)) & (gsmdata.times < pys.time_double(eet))])

        rmlatmltdata = pyt.get_data('erg_orb_l2_pos_rmlatmlt')
        mlat.extend(rmlatmltdata.y[:, 1][(rmlatmltdata.times > pys.time_double(sst)) & (rmlatmltdata.times < pys.time_double(eet))])
        mlt.extend(rmlatmltdata.y[:, 2][(rmlatmltdata.times > pys.time_double(sst)) & (rmlatmltdata.times < pys.time_double(eet))])


# %%
l_val = np.array(l_val).reshape(1,-1)[0]
gsmx = np.array(gsmx).reshape(1,-1)[0]
gsmy = np.array(gsmy).reshape(1,-1)[0]
gsmz = np.array(gsmz).reshape(1,-1)[0]
mlat = np.array(mlat).reshape(1,-1)[0]
mlt = np.array(mlt).reshape(1,-1)[0]

mlatmask = [(mlat < -10.) | (mlat > 10.)]

# %%
plt.rcParams["font.size"] = 20
fig = plt.figure(figsize=(25,5), facecolor='white')
ax1 = fig.add_subplot(151)
ax2 = fig.add_subplot(152)
ax3 = fig.add_subplot(153)
ax4 = fig.add_subplot(154)
ax5 = fig.add_subplot(155)


pys.erg.orb(trange=['2017-04-01 00:00:00','2017-04-01 8:59:59'])
gsmdata = pyt.get_data('erg_orb_l2_pos_gsm')
ax2.scatter(gsmdata.y[:,0], gsmdata.y[:,1], c='gray')
ax3.scatter(gsmdata.y[:,0], gsmdata.y[:,2], c='gray')
ax4.scatter(gsmdata.y[:,1], gsmdata.y[:,2], c='gray')

# pys.erg.orb(trange=['2017-05-31 00:00:00','2017-05-31 8:59:59'])
pys.erg.orb(trange=['2017-04-30 00:00:00','2017-04-30 8:59:59'])
gsmdata = pyt.get_data('erg_orb_l2_pos_gsm')
ax2.scatter(gsmdata.y[:,0], gsmdata.y[:,1], c='gray')
ax3.scatter(gsmdata.y[:,0], gsmdata.y[:,2], c='gray')
ax4.scatter(gsmdata.y[:,1], gsmdata.y[:,2], c='gray')


ax1.hist(l_val[mlatmask])
ax2.scatter(gsmx[mlatmask], gsmy[mlatmask])
ax3.scatter(gsmx[mlatmask], gsmz[mlatmask])
ax4.scatter(gsmy[mlatmask], gsmz[mlatmask])
ax5.scatter(np.sqrt(gsmx[mlatmask]**2+gsmy[mlatmask]**2), gsmz[mlatmask])

texx = ['Lm', 'gsmx', 'gsmx', 'gsmy', 'sqrt(gsmy^2+gsmz^2)']
texy = ['', 'gsmy', 'gsmz', 'gsmz', 'gsmz']
for i, axis in enumerate([ax1,ax2,ax3,ax4,ax5]):
    axis.set_xlabel(texx[i])
    axis.set_ylabel(texy[i])
    axis.grid()

for i, axis in enumerate([ax2,ax3,ax4]):
    axis.set_xlim(-6, 6)
    axis.set_ylim(-6, 6)
    axis.set_xticks([-6,-4,-2,0,2,4,6])
    axis.set_yticks([-6,-4,-2,0,2,4,6])
ax5.set_xticks([-6,-4,-2,0,2,4,6])
ax5.set_xlim(0, 6)
ax5.set_ylim(-6, 6)

# fig.title('2017/04-05')




fig.tight_layout()
plt.savefig('/Users/ampuku/Documents/duct/code/python/for_mlt_plot/event_201704', transparent=False)
# plt.savefig('/Users/ampuku/Documents/duct/code/python/for_mlt_plot/event_201704-05', transparent=False)

# %%

plt.rcParams["font.size"] = 15
fig = plt.figure(figsize=(5,5), facecolor='white')
ax1 = fig.add_subplot(111)


ax1.hist(l_val[mlatmask])
ax1.set_xlabel('Lm')
ax1.set_ylabel('')
ax1.grid()
fig.tight_layout()
plt.savefig('/Users/ampuku/Documents/duct/code/python/for_mlt_plot/event_201704-05_Lm', transparent=False)


# %%
# deg = mlt * 2*np.pi / 24
# Lx = l_val * np.cos(deg)
# Ly = l_val * np.sin(deg)

# # %%

# ax = fig.add_subplot(111)
# ax.scatter(Lx, Ly)
# ax.set_xlabel('')

# %%
plt.rcParams["font.size"] = 18
fig = plt.figure(figsize=(10,5), facecolor='white')
ax1 = fig.add_subplot(121,polar=True)
ax2 = fig.add_subplot(122)


pys.erg.orb(trange=['2017-04-01 00:00:00','2017-04-01 8:59:59'])
rmlatmltdata = pyt.get_data('erg_orb_l2_pos_rmlatmlt')
lmdata = pyt.get_data('erg_orb_l2_pos_Lm')
ax1.scatter(rmlatmltdata.y[:,2] * 2*np.pi / 24, lmdata.y[:,1], c='darkgrey', s=1, label='2017/04/01')
ax2.scatter(lmdata.y[:,1], rmlatmltdata.y[:,1], c='darkgrey', s=1)

# pys.erg.orb(trange=['2017-05-31 00:00:00','2017-05-31 8:59:59'])
pys.erg.orb(trange=['2017-04-30 00:00:00','2017-04-30 8:59:59'])
rmlatmltdata = pyt.get_data('erg_orb_l2_pos_rmlatmlt')
lmdata = pyt.get_data('erg_orb_l2_pos_Lm')
ax1.scatter(rmlatmltdata.y[:,2] * 2*np.pi / 24, lmdata.y[:,1], c='lightgray', s=1, label='2017/04/30')
# ax1.scatter(rmlatmltdata.y[:,2] * 2*np.pi / 24, lmdata.y[:,1], c='gray', s=1, label='2017-05-31')
ax2.scatter(lmdata.y[:,1], rmlatmltdata.y[:,1], c='lightgray', s=1)

mlatmask = [(mlat < -10.) | (mlat > 10.)]
deg = mlt * 2*np.pi / 24

ax1.scatter(deg[mlatmask],l_val[mlatmask],s=15,c='midnightblue')

ax1.set_rlim([0.0, 8.0])
ax1.set_rgrids([0,2,4,6,8],
                # labels=['','2','4','6',''],
                labels=['','2','4','6','  L='],
                # fontsize=12,
                angle=180) # angle で 表示方向を選択(度数法)

# theta方向の設定
ax1.set_thetalim([0.,2*np.pi])
# ラジアンではなく, 度数法で指定するっぽい
ax1.set_thetagrids([0, 90, 180, 270], 
                    # labels=["0", "6", "12", "18"])
                    labels=["0\nMLT", "6", "12", "18"])
                    # fontsize=12)
# ax.set_xlim([0, 24])
# ax.set_xticks([0,6,12,18])
# ax.set_xticklabels(["SW", "S", "SE", "E", "NE", "N", "NW", "W"])
# ax.set_ylim([0, 6.0])
# ax.set_yticks(np.arange(-3, 3.01, 1))
# ax.set_yticklabels(abc)

ax2.scatter(l_val[mlatmask], mlat[mlatmask],s=15,c='midnightblue')
ax2.set_xlim([2.0, 6.0])
ax2.set_xlabel('L値')
ax2.set_ylabel('MLAT')
ax2.grid()

ax1.legend()
fig.tight_layout()
plt.savefig('/Users/ampuku/Documents/duct/code/python/for_mlt_plot/event_mlt_201704', transparent=False, dpi=400)

# %%
