# %%
import numpy as np
import matplotlib.pyplot as plt
import pyspedas as pys
import pytplot

# %%

# メモ：要確認はまじで確認が必要、クリティカルなやつ　要対策はまあほっといても大丈夫なやつ


class imitation_tplot():
    def __init__(self, tt, y, v=None):
        self.times = tt
        self.y = y
        self.v = v


def mag_svd(time, data, nfft=8192, stride=4096, n_average=3):

    # scw = pyt.get_data('bfield')
    # efw = pyt.get_data('efield')
    # scw = imitation_tplot(tt, eb[:, 3:6])
    scw = imitation_tplot(time, data)

    # scw.y : 3次元*時間

    # データが入ってない場合の対策をしたい..要対策!!
    if 1:
        # if scw.y:
        #     print('No valid data are not available. Returning...')
        # if scw:

        # ===============================
        #  Perform FFTs
        # ===============================

        # nfftがwindowsize、strideがstrideに当たる
        nfft = nfft  # L(IDL) : long型  python3ではint型に最大最小の制限なし
        stride = stride

        ndata = len(scw.times)

        # 安福がpythonで書いてみた
        if 1:
            # 窓 np.hanning(nfft) を使って efw.y[j:j+nfft, k], scw.y[j:j+nfft, k] をフーリエ変換する
            # 1秒に65536サンプリングみたい、モードによっては8個に分けて処理する関係で65536/8=8192サンプリングになってるぽい？？栗田さんに要確認!!
            # t_e = efw.times[0] + (np.arange(i-1, dtype='float64')*stride+nfft/2) / 8192.
            # t_s = scw.times[0] + (np.arange(i-1, dtype='float64')*stride+nfft/2) / 8192.
            # freq = np.fft.fftfreq(nfft)の一部をとってくる

            scw_fft = np.empty((int((ndata-nfft)/stride)+1, int(nfft/2), 3), dtype='complex')

            win = np.hanning(nfft)
            win_cor = 1 / (sum(win)/nfft)

            i = 0
            t_s = []
            for j in range(0, ndata-nfft+1, stride):
                scw_fft[i, :, 0] = np.fft.fft(scw.y[j:j+nfft, 0] * win)[:int(nfft/2)] * win_cor
                scw_fft[i, :, 1] = np.fft.fft(scw.y[j:j+nfft, 1] * win)[:int(nfft/2)] * win_cor
                scw_fft[i, :, 2] = np.fft.fft(scw.y[j:j+nfft, 2] * win)[:int(nfft/2)] * win_cor

                t_s.append(scw.times[j+int(nfft/2)])

                i += 1

            freq = np.fft.fftfreq(nfft, d=scw.times[2]-scw.times[1])[:int(nfft/2)]

            scw_fft_tot = np.sqrt(np.abs(scw_fft[:, :, 0])**2 + np.abs(scw_fft[:, :, 1])**2 + np.abs(scw_fft[:, :, 2])**2) * 1E-12
            pytplot.store_data('b_total', data={'x': t_s, 'y': scw_fft_tot, 'v': freq})
            pytplot.options('b_total', option='spec', value=3)
            pytplot.options('b_total', option='ylog', value=1)
            pytplot.options('b_total', option='yrange', value=[32, 20000])
            pytplot.options('b_total', option='zrange', value=[1e-8, 1e0])

# ===============================
#  Magnetic SVD analysis
# ===============================

        wna = scw_fft_tot*0.0
        phi = scw_fft_tot*0.0
        plan = wna
        elip = wna
        # vcc = 3.0e8
        # counter_start = 0.0
        # print(' ')
        # print('Total Number of steps:'+str())
        # print(' ')

        npt = scw_fft.shape[0]

        for i in range(0, npt):
            # 何% 計算できたかを表示する ひとまず飛ばす 要検討
            # if 10*float(i)/(npt-1) > (counter_start+1):
            #     print(str(100*float(i)/(npt-1), 2) + ' % Complete ')
            #     print(' Processing step no. :' + str(i+1, 2))
            #     counter_start += 1

            for j in range(0, len(freq)):
                # making spectral matrix

                Atmp = np.empty((6, 3), dtype='complex128')

                z = scw_fft[i, j, :]
                Atmp[0, :] = np.real(z[0] * np.conj(z))
                Atmp[1, :] = np.real(z[1] * np.conj(z))
                Atmp[2, :] = np.real(z[2] * np.conj(z))
                Atmp[3, :] = np.imag([0.0, -z[0]*np.conj(z[1]), -z[0]*np.conj(z[2])])
                Atmp[4, :] = np.imag([z[0]*np.conj(z[1]), 0.0, -z[1]*np.conj(z[2])])
                Atmp[5, :] = np.imag([z[0]*np.conj(z[2]), z[1]*np.conj(z[2]), 0.0])

                A = Atmp

                u, w, v = np.linalg.svd(A)

# ===============================
#  Polarization calculation
# ===============================

                if w[-1] > 0.:
                    # planarity
                    plan[i, j] = 1. - np.sqrt(w[0]/w[2])

                    # ellipticity with the sign of polarization sense
                    if np.imag(scw_fft[i, j, 0]*np.conj(scw_fft[i, j, 1])) < 0:
                        elip[i, j] = - w[1] / w[2]
                    else:
                        elip[i, j] = w[1] / w[2]

                    # 栗田さんコメント
                    # magnetic SVD cannot reveal the sign of k-vector, so the k-vector direction is changed to have kz > 0.
                    if v[0, 2] < 0.0:
                        v[0, :] = - 1.0 * v[0, :]

                    # polar angle of k-vector
                    wna[i, j] = np.arctan(np.sqrt(v[0, 0]**2 + v[0, 1]**2) / v[0, 2]) / (np.pi/180)

                    # azimuth angle of k-vector
                    if v[0, 0] >= 0:
                        phi[i, j] = np.arctan(v[0, 1]/v[0, 0]) / (np.pi/180)
                    if v[0, 0] < 0 and v[0, 1] < 0.0:
                        phi[i, j] = np.arctan(v[0, 1]/v[0, 0]) / (np.pi/180) - 180.0
                    if v[0, 0] < 0 and v[0, 1] >= 0.0:
                        phi[i, j] = np.arctan(v[0, 1]/v[0, 0]) / (np.pi/180) + 180.0

        pytplot.store_data('waveangle_th_magsvd', data={'x': t_s, 'y': wna, 'v': freq})
        pytplot.store_data('waveangle_phi_magsvd', data={'x': t_s, 'y': phi, 'v': freq})
        pytplot.store_data('planarity_magsvd', data={'x': t_s, 'y': plan, 'v': freq})
        waveangle_th_magsvd = imitation_tplot(t_s, wna, freq)
        waveangle_phi_magsvd = imitation_tplot(t_s, phi, freq)
        planarity_magsvd = imitation_tplot(t_s, plan, freq)

        # return
        return waveangle_th_magsvd, waveangle_phi_magsvd, planarity_magsvd

    else:
        print('データが入ってなかった...')
        return

# %%
