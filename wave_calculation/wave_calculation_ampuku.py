# %%
import pytplot
import pyspedas as pys
import numpy as np
from model_waveform_01 import model_waveform_class
from mag_svd_test import mag_svd


# %%
model1 = model_waveform_class(th00=20)
model2 = model_waveform_class(th00=40)

model12_eb = model1.eb + model2.eb
model12_eb = np.transpose(model12_eb)

waveangle_th_magsvd, waveangle_phi_magsvd, planarity_magsvd = mag_svd(model1.times, model12_eb)


# %%

time_range = ['2017-07-03/04:09:18', '2017-07-03/04:10:07']

pys.erg.pwe_wfc(trange=time_range, datatype='waveform', mode='65khz', level='l2')
pytplot.tplot(['erg_pwe_wfc_l2_b_65khz_Bx_waveform', 'erg_pwe_wfc_l2_b_65khz_By_waveform', 'erg_pwe_wfc_l2_b_65khz_Bz_waveform'])

# %%
Bx_time, Bx_data = pytplot.get_data('erg_pwe_wfc_l2_b_65khz_Bx_waveform')
By_time, By_data = pytplot.get_data('erg_pwe_wfc_l2_b_65khz_By_waveform')
Bz_time, Bz_data = pytplot.get_data('erg_pwe_wfc_l2_b_65khz_Bz_waveform')

# pytplot.store_data('test1', data={'x':Bx_time, 'y':Bx_data})

data = np.empty((len(Bx_time), 3))
data[:, 0] = Bx_data
data[:, 1] = By_data
data[:, 2] = Bz_data

mag_svd(Bx_time, data, nfft=4096, stride=2048, n_average=1)


# %%
pytplot.options(['waveangle_th_magsvd', 'waveangle_phi_magsvd', 'planarity_magsvd'], option='spec', value=3)
pytplot.options(['waveangle_th_magsvd', 'waveangle_phi_magsvd', 'planarity_magsvd'], option='ylog', value=1)
pytplot.options(['waveangle_th_magsvd', 'waveangle_phi_magsvd', 'planarity_magsvd'], option='ylim', value=[50, 20000])
pytplot.tplot(['waveangle_th_magsvd', 'waveangle_phi_magsvd', 'planarity_magsvd'])


# %%
