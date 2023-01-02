# %%
import numpy as np
import matplotlib.pyplot as plt
import pyspedas as pys
import pytplot

# %%
# import pandas as pd
# path = '/Users/ampuku/Documents/duct/code/python/wave_calculation/data/'
# time = pd.read_csv(path + 'wfc_time.txt', header=None)
# bx_waveform = pd.read_csv(path + 'wfc_bx_waveform.txt', header=None)
# by_waveform = pd.read_csv(path + 'wfc_by_waveform.txt', header=None)
# bz_waveform = pd.read_csv(path + 'wfc_bz_waveform.txt', header=None)
# nfft=4096; stride=2048; n_average=3; tplot=0

# %%


class data_form():
    def __init__(self, time, y, v=None):
        self.times = time
        self.y = y
        self.v = v


def mag_svd(time, bx_waveform, by_waveform, bz_waveform, nfft=4096, stride=2048, t_average=3, f_average=3, tplot=0):

    # ===============================
    #  FFT
    # ===============================
    time = np.array(time).reshape(-1)
    bx_waveform = np.array(bx_waveform).reshape(-1)
    by_waveform = np.array(by_waveform).reshape(-1)
    bz_waveform = np.array(bz_waveform).reshape(-1)

    nfft = nfft
    stride = stride
    ndata = len(time)
    fsamp = 1. / (time[2]-time[1])
    npt = int((ndata-nfft)/stride)

    scw_fft = np.empty((int((ndata-nfft)/stride)+1, int(nfft/2), 3), dtype='complex')

    win = np.hanning(nfft)*8/3
    # win_cor = 1 / (sum(win)/nfft)

    i = 0
    t_s = []
    for j in range(0, ndata-nfft+1, stride):
        scw_fft[i, :, 0] = np.fft.fft(bx_waveform[j:j+nfft] * win)[:int(nfft/2)]
        scw_fft[i, :, 1] = np.fft.fft(by_waveform[j:j+nfft] * win)[:int(nfft/2)]
        scw_fft[i, :, 2] = np.fft.fft(bz_waveform[j:j+nfft] * win)[:int(nfft/2)]

        i += 1
    t_s = time[0]+(np.arange(i-1)*stride+nfft/2) / fsamp

    freq = np.fft.fftfreq(nfft, d=time[2]-time[1])[:int(nfft/2)]
    bw = fsamp/nfft

    scw_fft_tot = np.abs(scw_fft[0:npt, 0:int(nfft/2), 0])**2 / bw + np.abs(scw_fft[0:npt, 0:int(nfft/2), 1])**2 / \
        bw + np.abs(scw_fft[0:npt, 0:int(nfft/2), 2])**2 / bw


# ===============================
#  Magnetic SVD analysis
# ===============================

    wna = scw_fft_tot*0.0
    phi = scw_fft_tot*0.0
    planarity = scw_fft_tot*0.0
    polarization = scw_fft_tot*0.0
    bspec = scw_fft_tot*0.0
    # counter_start = 0.0

    npt = (ndata-nfft)/stride-1

    for i in range(0, int(npt-t_average)):

        # spectral matrixを作成

        index_t = i + np.arange(t_average)

        bubu = np.sum(scw_fft[index_t, :, 0]*np.conj(scw_fft[index_t, :, 0]), axis=0) / t_average
        bubv = np.sum(scw_fft[index_t, :, 0]*np.conj(scw_fft[index_t, :, 1]), axis=0) / t_average
        bubw = np.sum(scw_fft[index_t, :, 0]*np.conj(scw_fft[index_t, :, 2]), axis=0) / t_average

        bvbv = np.sum(scw_fft[index_t, :, 1]*np.conj(scw_fft[index_t, :, 1]), axis=0) / t_average
        bvbw = np.sum(scw_fft[index_t, :, 1]*np.conj(scw_fft[index_t, :, 2]), axis=0) / t_average

        bwbw = np.sum(scw_fft[index_t, :, 2]*np.conj(scw_fft[index_t, :, 2]), axis=0) / t_average

        for j in range(int((f_average-1)/2), len(freq)-int((f_average+1)/2)):

            index_f = j + np.arange(f_average)
            A = np.array([[np.sum(np.real(bubu[index_f])),      np.sum(np.real(bubv[index_f])),      np.sum(np.real(bubw[index_f]))],
                          [np.sum(np.real(bubv[index_f])),      np.sum(np.real(bvbv[index_f])),      np.sum(np.real(bvbw[index_f]))],
                          [np.sum(np.real(bubw[index_f])),      np.sum(np.real(bvbw[index_f])),      np.sum(np.real(bwbw[index_f]))],
                          [0.0,     np.sum(-np.imag(bubv[index_f])),     np.sum(-np.imag(bubw[index_f]))],
                          [np.sum(np.imag(bubv[index_f])),      0.0,     np.sum(-np.imag(bvbw[index_f]))],
                          [np.sum(np.imag(bubw[index_f])),      np.sum(np.imag(bvbw[index_f])),      0.0]])

            bspec[i, j] = np.sqrt(A[0, 0]**2 + A[1, 1]**2 + A[2, 2]**2) / f_average

            # SVD 特異値分解を実行
            u, w, v = np.linalg.svd(A)

            if w[0] > 0.:
                # planarity
                planarity[i, j] = 1. - np.sqrt(w[2]/w[0])

                # polarization
                if np.imag(scw_fft[i, j, 0]*np.conj(scw_fft[i, j, 1])) < 0:
                    polarization[i, j] = - w[1] / w[0]
                else:
                    polarization[i, j] = w[1] / w[0]

                if v[2, 2] < 0.0:
                    v[2, :] = - 1.0 * v[2, :]

                # WNA
                wna[i, j] = np.arctan(np.sqrt(v[2, 0]**2 + v[2, 1]**2) / v[2, 2]) / (np.pi/180)

                # 方位角方向の伝搬角
                if v[2, 0] >= 0:
                    phi[i, j] = np.arctan(v[2, 1]/v[2, 0]) / (np.pi/180)
                if v[2, 0] < 0 and v[2, 1] < 0.0:
                    phi[i, j] = np.arctan(v[2, 1]/v[2, 0]) / (np.pi/180) - 180.0
                if v[2, 0] < 0 and v[2, 1] >= 0.0:
                    phi[i, j] = np.arctan(v[2, 1]/v[2, 0]) / (np.pi/180) + 180.0

    # if tplot == 0:
    #     bspec = data_form(time=t_s, y=scw_fft_tot, v=freq)

    # if tplot == 1:
    #     pytplot.store_data('bspec', data={'x': t_s, 'y': scw_fft_tot, 'v': freq})
    #     pytplot.options('bspec', option='spec', value=3)
    #     pytplot.options('bspec', option='ylog', value=1)
    #     pytplot.options('bspec', option='yrange', value=[32, 20000])
    #     pytplot.options('bspec', option='zlog', value=1)
    #     # pytplot.options('bspec', option='zrange', value=[1e-4, 1e2])
    #     pytplot.options('bspec', option='zrange', value=[1e2, 1e6])
    #     pytplot.options('bspec', option='zsubtitle', value='pT^2/Hz')

    if tplot == 0:
        waveangle_th_magsvd = data_form(t_s, wna, freq)
        waveangle_phi_magsvd = data_form(t_s, phi, freq)
        planarity_magsvd = data_form(t_s, planarity, freq)
        polarization_magsvd = data_form(t_s, polarization, freq)

        return bspec, waveangle_th_magsvd, waveangle_phi_magsvd, polarization_magsvd, planarity_magsvd

    if tplot == 1:
        pytplot.store_data('waveangle_th_magsvd', data={'x': t_s, 'y': wna, 'v': freq})
        pytplot.store_data('waveangle_phi_magsvd', data={'x': t_s, 'y': phi, 'v': freq})
        pytplot.store_data('planarity_magsvd', data={'x': t_s, 'y': planarity, 'v': freq})
        pytplot.store_data('polarization_magsvd', data={'x': t_s, 'y': polarization, 'v': freq})

        pytplot.options(['waveangle_th_magsvd'], option='zrange', value=[0., 90.])
        pytplot.options(['waveangle_phi_magsvd'], option='zrange', value=[-180., 180.])
        pytplot.options(['planarity_magsvd'], option='zrange', value=[0., 1.])
        pytplot.options(['polarization_magsvd'], option='zrange', value=[-1., 1.])
        pytplot.options(['waveangle_th_magsvd', 'waveangle_phi_magsvd', 'polarization_magsvd', 'planarity_magsvd'], option='spec', value=3)
        pytplot.options(['waveangle_th_magsvd', 'waveangle_phi_magsvd', 'polarization_magsvd',
                        'planarity_magsvd'], option='ysubtitle', value='frequency Hz')
        pytplot.options(['waveangle_th_magsvd', 'waveangle_phi_magsvd'], option='zsubtitle', value='degree')

        return

# %%
