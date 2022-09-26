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
