# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %%
fce_ave_all = [14.105383, 13.445186, 30.402452, 14.299986]  # 規格化用

# %%
plt.rcParams["font.size"] = 25
fig = plt.figure(figsize=(25, 8), facecolor='white')

path = '/Users/ampuku/Documents/duct/code/python/for_M_thesis/event1/'

fdata = pd.read_csv(path+'f.csv', header=None)
phidata = pd.read_csv(path+'phi.csv', header=None)
fce_ave = fce_ave_all[0]


ax1 = fig.add_subplot(1, 3, 1)
ax1.scatter(fdata/fce_ave, phidata, color='k', linewidth=2)
ax1.set_xlim(0.05, 0.30)
ax1.set_xlabel("f / fc")
# ax1.set_ylabel("phi [degree]")
# ax1.set_xticks([1.,1.5,2.,2.5,3.,3.5,4.])
# ax1.set_xticklabels([])
ax1.set_title('(a) Event1\n', fontsize=30)


path = '/Users/ampuku/Documents/duct/code/python/for_M_thesis/event2/'

fdata = pd.read_csv(path+'f.csv', header=None)
phidata = pd.read_csv(path+'phi.csv', header=None)
fce_ave = fce_ave_all[1]


ax2 = fig.add_subplot(1, 3, 2)
ax2.scatter(fdata/fce_ave, phidata, color='k', linewidth=2)
ax2.set_xlim(0.25, 0.5)
ax2.set_xlabel("f / fc")
# ax2.set_ylabel("phi [degree]")
# ax2.set_ylim(0., 0.1)
# ax2.set_xlabel("frequency [kHz]")
# ax2.set_xlabel("f / fc")
ax2.tick_params(labelbottom=True, labelleft=False, labelright=False, labeltop=False)
ax2.set_title('(b) Event2\n', fontsize=30)


path = '/Users/ampuku/Documents/duct/code/python/for_M_thesis/event3/'

fdata = pd.read_csv(path+'f.csv', header=None)
phidata = pd.read_csv(path+'phi.csv', header=None)
fce_ave = fce_ave_all[2]


ax3 = fig.add_subplot(1, 3, 3)
ax3.scatter(fdata/fce_ave, phidata, color='k', linewidth=2)
ax3.set_xlim(0.05, 0.15)
# ax3.set_xlim(0.02, 0.17)
# ax3.set_ylim(0., 0.1)
# ax3.set_xlabel("frequency [kHz]")
# ax3.set_xlabel("f / fc")
ax3.set_xlabel("f / fc")
# ax3.set_ylabel("phi [degree]")
ax3.tick_params(labelbottom=True, labelleft=False, labelright=False, labeltop=False)
ax3.set_title('(c) Event3\n', fontsize=30)

for ax in [ax1, ax2, ax3]:
    ax.grid()

fig.tight_layout()

fig.savefig('/Users/ampuku/Desktop/test.png', dpi=300)


# %%

# %%
