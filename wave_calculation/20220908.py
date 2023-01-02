# %%
import matplotlib.pyplot as plt
import numpy as np
import pyspedas as pys
import pytplot

# %%
# get data
time_range = ['2017-07-03/04:09:18', '2017-07-03/04:10:07']
pys.erg.pwe_wfc(trange=time_range, datatype='waveform', mode='65khz', level='l2')
pytplot.tplot(['erg_pwe_wfc_l2_b_65khz_Bx_waveform', 'erg_pwe_wfc_l2_b_65khz_By_waveform', 'erg_pwe_wfc_l2_b_65khz_Bz_waveform'])

Bx_time, Bx_data = pytplot.get_data('erg_pwe_wfc_l2_b_65khz_Bx_waveform')
By_time, By_data = pytplot.get_data('erg_pwe_wfc_l2_b_65khz_By_waveform')
Bz_time, Bz_data = pytplot.get_data('erg_pwe_wfc_l2_b_65khz_Bz_waveform')

# %%
data = np.empty((len(Bx_data), 3))

data[:, 0] = Bx_data
data[:, 1] = By_data
data[:, 2] = Bz_data

# %%
from mag_svd_test import mag_svd
mag_svd(Bx_time, data, nfft=8192, stride=4096, n_average=3)
# %%
pytplot.tplot_names()

pytplot.options(['waveangle_th_magsvd','waveangle_phi_magsvd','planarity_magsvd'], option='Spec', value=1)
pytplot.options(['waveangle_th_magsvd','waveangle_phi_magsvd','planarity_magsvd'], option='ylog', value=1)
pytplot.options(['waveangle_th_magsvd','waveangle_phi_magsvd','planarity_magsvd'], option='yrange', value=[32,20000])

pytplot.tplot(['waveangle_th_magsvd','waveangle_phi_magsvd','planarity_magsvd'])
# %%
