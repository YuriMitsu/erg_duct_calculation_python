# %%
import numpy as np
import matplotlib.pyplot as plt
import pyspedas as pys
import pytplot as pyt

# %%

# メモ：要確認はまじで確認が必要、クリティカルなやつ　要対策はまあほっといても大丈夫なやつ


class imitation_tplot():
    def __init__(self, tt, y, v=None):
        self.times = tt
        self.y = y
        self.v = v


def me_svd(tt, eb):

    # scw = pyt.get_data('bfield')
    # efw = pyt.get_data('efield')
    scw = imitation_tplot(tt, eb[:, 3:6])
    efw = imitation_tplot(tt, eb[:, 0:3])

    # scw.y : 3次元*時間

    # データが入ってない場合の対策をしたい..要対策!!
    if 1:
        # if scw.y:
        #     print('No valid data are not available. Returning...')
        # if scw:

        # ===============================
        #  Perform FFTs
        # ===============================

        nfft = 8192  # L(IDL) : long型  python3ではint型に最大最小の制限なし
        stride = nfft

        ndata = len(scw.times)

        # 安福がpythonで書いてみた
        if 1:
            # 窓 np.hanning(nfft) を使って efw.y[j:j+nfft, k], scw.y[j:j+nfft, k] をフーリエ変換する
            # 1秒に65536サンプリングみたい、モードによっては8個に分けて処理する関係で65536/8=8192サンプリングになってるぽい？？栗田さんに要確認!!
            # t_e = efw.times[0] + (np.arange(i-1, dtype='float64')*stride+nfft/2) / 8192.
            # t_s = scw.times[0] + (np.arange(i-1, dtype='float64')*stride+nfft/2) / 8192.
            # freq = np.fft.fftfreq(nfft)の一部をとってくる

            efw_fft = np.empty((int((ndata-nfft)/stride)+1, int(nfft/2), 3), dtype='complex')
            scw_fft = np.empty((int((ndata-nfft)/stride)+1, int(nfft/2), 3), dtype='complex')

            win = np.hanning(nfft)
            win_cor = 1 / (sum(win)/nfft)

            i = 0
            t_e = []
            t_s = []
            for j in range(0, ndata-nfft+1, stride):
                efw_fft[i, :, 0] = np.fft.fft(efw.y[j:j+nfft, 0] * win)[:int(nfft/2)] / win_cor
                efw_fft[i, :, 1] = np.fft.fft(efw.y[j:j+nfft, 1] * win)[:int(nfft/2)] / win_cor
                efw_fft[i, :, 2] = np.fft.fft(efw.y[j:j+nfft, 2] * win)[:int(nfft/2)] / win_cor

                scw_fft[i, :, 0] = np.fft.fft(scw.y[j:j+nfft, 0] * win)[:int(nfft/2)] / win_cor
                scw_fft[i, :, 1] = np.fft.fft(scw.y[j:j+nfft, 1] * win)[:int(nfft/2)] / win_cor
                scw_fft[i, :, 2] = np.fft.fft(scw.y[j:j+nfft, 2] * win)[:int(nfft/2)] / win_cor

                t_e.append(efw.times[j+int(nfft/2)])
                t_s.append(scw.times[j+int(nfft/2)])

                i += 1

            freq = np.fft.fftfreq(nfft)[:int(nfft/2)]

            efw_fft_tot = np.abs(efw_fft[:, :, 0])**2 + np.abs(efw_fft[:, :, 1])**2 + np.abs(efw_fft[:, :, 2])**2
            scw_fft_tot = np.abs(scw_fft[:, :, 0])**2 + np.abs(scw_fft[:, :, 1])**2 + np.abs(scw_fft[:, :, 2])**2

        # 栗田さんIDLコードそのまま
        if 0:

            efw_fft = np.empty(
                (int((ndata-nfft)/stride)+1, nfft, 3), dtype='complex')
            # efw_fft=dcomplexarr(long(ndata-nfft)/stride+1,nfft,3)
            scw_fft = np.empty(
                (int((ndata-nfft)/stride)+1, nfft, 3), dtype='complex')
            # scw_fft=dcomplexarr(long(ndata-nfft)/stride+1,nfft,3)
            win = np.hanning(nfft)*8./3.

            i = 0
            for j in range(0, ndata-nfft, stride):  # range(a,b,c)はaからb-1までc刻みで数値を返す
                for k in [0, 1, 2]:
                    efw_fft[i, :, k] = np.fft.fft(
                        efw.y[j:j+nfft, k]*win)  # 配列指定最後は入らない
                for k in [0, 1, 2]:
                    scw_fft[i, :, k] = np.fft.fft(scw.y[j:j+nfft, k]*win)
                i += 1

            # 最後の一要素をカウントしてないのはなぜ？
            npt = len(scw_fft[0:int((ndata-nfft)/stride), 0, 0])
            t_e = efw.times[0] + (np.arange(i-1, dtype='float64')*stride+nfft/2) / \
                8192.
            t_s = scw.times[0] + (np.arange(i-1, dtype='float64')*stride+nfft/2) / \
                8192.
            freq = np.arange(nfft/2, dtype='float64')*8192/nfft
            bw = 8192 / nfft
            efw_fft_tot = np.float64(abs(efw_fft[0:npt, 0:int(nfft/2), 0])**2/bw +
                                     abs(efw_fft[0:npt, 0:int(nfft/2), 1])**2/bw+abs(efw_fft[0:npt, 0:int(nfft/2), 2])**2/bw)
            scw_fft_tot = np.float64(abs(scw_fft[0:npt, 0:int(nfft/2), 0])**2/bw +
                                     abs(scw_fft[0:npt, 0:int(nfft/2), 1])**2/bw+abs(scw_fft[0:npt, 0:int(nfft/2), 2])**2/bw)

            # efwlim = {spec: 1, zlog: 1, ylog: 0, yrange: [100, 4096], ystyle: 1}
            # scwlim = {spec: 1, zlog: 1, ylog: 0, yrange: [100, 4096], ystyle: 1}

            # pyt.store_data('efw_fft_x')
            # pyt.store_data('efw_fft_y')
            # pyt.store_data('efw_fft_z')
            # pyt.store_data('efw_fft_all')

            # pyt.store_data('sfw_fft_x')
            # pyt.store_data('sfw_fft_y')
            # pyt.store_data('sfw_fft_z')
            # pyt.store_data('sfw_fft_all')


# ===============================
#  Electromagnetic SVD analysis
# ===============================

        wna = scw_fft_tot*0.0
        phi = scw_fft_tot*0.0
        plan = wna
        vcc = 3.0e8
        counter_start = 0.0
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
                # making spectral matrix (first version, no emsemble average...)
                z = np.array([vcc*scw_fft[i, j, 0], vcc*scw_fft[i, j, 1], vcc*scw_fft[i, j, 2],
                        efw_fft[i, j, 0], efw_fft[i, j, 1], efw_fft[i, j, 2]])


"""
ここから！！！
Atmpがなんで3*18なのか？！
→ これ、em SVDだった...
私がひとまずやりたいのはMagnetic SVD...
ここまでで置いておいて、Magnetic SVDに移ろう...
"""

                Atmp = np.empty((3, 18), dtype='complex128')
                Btmp = np.empty((18), dtype='complex128')

                for ii in range(1, 7):
                    Atmp[:, (ii-1)*3:ii*3-1] = np.array([[0.0, z[5], -z[4]], [-z[5],
                        0.0, z[3]], [z[4], -z[3], 0.0]], dtype=np.complex128)*np.conjugate(z[ii-1])
                    Btmp[(ii-1)*3:ii*3-1] = np.array([z[0], z[1], z[2]], dtype=np.complex128) * \
                        np.conjugate(z[ii-1])

                A = 0.0
                B = 0.0

                A = np.concatenate(np.transpose(np.real(Atmp)), np.transpose(np.imaginary(Atmp)))
                B = np.concatenate(np.real(Atmp), np.imaginary(Atmp))

                u, w, v = np.linalg.svd(np.transpose(A))

# Calculation of refractive index

                sv = np.empty(3, 3)
                sv[0, 0] = w[0]
                sv[1, 1] = w[1]
                sv[2, 2] = w[2]
                k = v.reshape(-1, 1) * np.linalg.inv(sv) * \
                    np.transpose(u) * np.transpose(B)
                # k=v ## invert(sv) ## transpose(u) ## transpose(B)
                beta_mat = A.reshape(-1, 1) * k
                k = k / np.sqrt(k[0]**2.+k[1]**2.+k[2]**2.)


# ===============================
#  Polarization calculation
# ===============================

                if min(w) > 0.:
                    # k-vector (perp to polarization plane) direction
                    wna[i, j] = np.arctan(
                        np.sqrt(k[0]**2+k[1]**2), k[2]) / (np.pi/180)

                    # azimuth angle of k-vector
                    if k[0] >= 0:
                        phi[i, j] = np.arctan(k[1]/k[0]) / (np.pi/180)
                    if k[0] < 0 and k[1] < 0.0:
                        phi[i, j] = np.arctan(k[1]/k[0]) / (np.pi/180) - 180.0
                    if k[0] < 0 and k[1] >= 0.0:
                        phi[i, j] = np.arctan(k[1]/k[0]) / (np.pi/180) + 180.0

                    # Electromagnetic planarity
                    n_plan = 0.0
                    d_plan = n_plan

                    for ijk in range(0, 35):
                        n_plan = n_plan + (beta_mat[ijk]-B[ijk])**2
                        d_plan = d_plan + \
                            (np.abs(beta_mat[ijk])+np.abs(B[ijk]))**2

                    plan[i, j] = 1.0-np.sqrt(n_plan/d_plan)

# Storing data into tplot vars.
        # wnalim={spec:1,zlog:0,ylog:0,yrange:[100,4096],ystyle:1,zrange:[0.0,180.0]}
        # philim={spec:1,zlog:0,ylog:0,yrange:[100,4096],ystyle:1,zrange:[-180.0,180.0]}
        # planlim={spec:1,zlog:0,ylog:0,yrange:[100,4096],ystyle:1,zrange:[0.0,1.0]}

        # store_data,'waveangle_th_emsvd',data={x:t_s,y:wna,v:freq},dlim=wnalim
        # store_data,'waveangle_phi_emsvd',data={x:t_s,y:phi,v:freq},dlim=philim
        # store_data,'planarity_emsvd',data={x:t_s,y:plan,v:freq},dlim=planlim

        waveangle_th_emsvd = imitation_tplot(t_s, wna, freq)
        waveangle_phi_emsvd = imitation_tplot(t_s, phi, freq)
        planarity_emsvd = imitation_tplot(t_s, plan, freq)

        return waveangle_th_emsvd, waveangle_phi_emsvd, planarity_emsvd

    else:
        print('データが入ってなかった...')
        return
