# %%
import numpy as np
import matplotlib.pylab as plt
import pyspedas as ps
import pytplot

# %%
# param00
# idata = 2048
# th00 = 30  # WNA [deg]
# ww = 0.3   # [fce]
# fpc = 4.0   # fp/fce
# bb = 5.e-7   # [nT]
# dsmp = 1./65.536e3  # WFCのサンプリングレート
# pi = np.pi
# cc = 2.99792458e8  # 光速[m/s]
# qq = 1.602e-19  # 電気素量
# kb = 1.38e-23  # よく見るkb ボルツマン定数
# me = 9.1e-31  # 電子質量
# ev = 1.16e4  # 1[eV] = 1.16e4[K]
# ipol = +1  # +1 : R-mode  -1 : L-mode

# %%

class model_waveform_class:

    def __init__(self, th00=30):
        self.idata = 2048*64
        # self.th00 = 30  # WNA [deg]
        self.th00 = th00  # WNA [deg]
        self.ww = 0.3   # [fce]
        self.fpc = 4.0   # fp/fce
        self.bb = 5.e-7   # [nT]
        self.dsmp = 1./65.536e3  # WFCのサンプリングレート
        self.pi = np.pi
        # self.cc = 2.99792458e8  # 光速[m/s]
        self.qq = 1.602e-19  # 電気素量
        self.kb = 1.38e-23  # よく見るkb ボルツマン定数
        self.me = 9.1e-31  # 電子質量
        self.ev = 1.16e4  # 1[eV] = 1.16e4[K]
        self.ipol = +1  # +1 : R-mode  -1 : L-mode

        self.main_model_waveform()

    def main_model_waveform(self):

        dsmp = self.dsmp
        idata = self.idata
        pi = self.pi

        # 1-3: Ex-Ez  4-6: Bx-Bz  B0=(0,0,1)

        eb = self.model_waveform()
        self.eb = eb

        Bx_max = 0.e0
        By_max = 0.e0

        Bx_max = max(eb[4, :])
        By_max = max(eb[5, :])
        print('(2(1PE12.4))', Bx_max, By_max)
        Br0 = 0.5e0*(Bx_max + By_max)
        Bl0 = abs(0.5e0*(Bx_max - By_max))
        print('(4(1PE12.4))', Br0, Bl0, Br0/Br0, Bl0/Br0)

        tt = []
        Br = []
        Bl = []
        ang00 = []
        ang00_r = []

        for ii in range(0, idata, 1):
            tt.append(float(ii) * dsmp)
            Bx1 = eb[4, ii]
            By1 = eb[5, ii]
            Br.append([Bx1*Br0/Bx_max, By1*Br0/By_max])
            Bl.append([Bx1*Bl0/Bx_max, -By1*Bl0/By_max])

            bw_perp = np.sqrt(Bx1**2 + By1**2)
            ang00_ = np.arccos(Bx1/bw_perp)
            ang00_ = ang00_/pi*180.e0
            if By1 < 0.e0:
                ang00_ = 360.e0 - ang00_
            ang00.append(ang00_)

            bw_perp_r = np.sqrt(Br[ii][0]**2 + Br[ii][1]**2)
            ang00_r_ = np.arccos(Br[ii][0]/bw_perp_r)
            ang00_r_ = ang00_r_/pi*180.e0
            if Br[ii][1] < 0.e0:
                ang00_r_ = 360.e0 - ang00_r_
            ang00_r.append(ang00_r_)

        self.times = tt
        self.Br = Br
        self.Bl = Bl
        self.ang00 = ang00
        self.ang00_r = ang00_r

        return
        # return tt, eb, Br, Bl, ang00, ang00_r

    def model_waveform(self):

        idata = self.idata
        th00 = self.th00
        ww = self.ww
        fpc = self.fpc
        dsmp = self.dsmp
        pi = self.pi

        wce = 1.e4   # fce [Hz]
        wce = wce*2.e0*pi

        ck, ref_n, vph, iflg = self.AH_rout()
        if iflg == 1:
            print(6, '(11Hno solution)')
            stop = input("何かがへんなのでstop")
        print('params : '+str(ww)+' [W/Wce] ,'+str(ck)+' [kR]')
        print('Phase Velocity     = '+str(vph)+' [c]')
        print('Refractive index   = '+str(ref_n)+' ')

        th = th00*pi/180.e0

        xx = (fpc/ww)**2
        yy = 1.e0/ww

        Ey_Ex = xx*yy/(yy**2-1.e0)/(xx/(yy**2-1.e0)+1.e0-ref_n**2)
        Ex_Ez = (xx-1.e0+ref_n**2*np.sin(th)**2) / \
            (ref_n*np.sin(th)*np.cos(th))**2
        Ey_Ez = Ey_Ex*Ex_Ez

        Ey_Ex = abs(Ey_Ex)
        Ex_Ez = abs(Ex_Ez)
        Ey_Ez = abs(Ey_Ez)

        Ex00 = 1.e0/ref_n
        Ey00 = Ey_Ex/ref_n
        Ez00 = 1.e0/Ex_Ez/ref_n

        Bx00 = -np.cos(th)*Ey00
        By00 = np.cos(th)*Ex00 - np.sin(th)*Ez00
        Bz00 = np.sin(th)*Ey00

        eb = np.ndarray([6, idata])
        for ii in range(0, idata, 1):
            tt = float(ii)*dsmp
            phs = ww*wce*tt
            Ex = Ex00*np.cos(phs)
            Ey = Ey00*np.sin(phs)
            Ez = Ez00*np.cos(phs)
            Bx = Bx00*np.sin(phs)
            By = By00*np.cos(phs)
            Bz = Bz00*np.sin(phs)

            bw_perp = np.sqrt(Bx**2 + By**2)
            ang = np.arccos(Bx/bw_perp)
            ang = ang/pi*180.e0
            if By < 0.e0:
                ang = 360.e0 - ang

            Ex1 = Ex*np.cos(th) - Ez*np.sin(th)
            Ey1 = Ey
            Ez1 = -Ex*np.sin(th) + Ez*np.cos(th)
            Bx1 = Bx*np.cos(th) - Bz*np.sin(th)
            By1 = By
            Bz1 = Bx*np.sin(th) + Bz*np.cos(th)
            bw_perp = np.sqrt(Bx1**2 + By1**2)
            ang00 = np.arccos(Bx1/bw_perp)
            ang00 = ang00/pi*180.e0
            if By1 < 0.e0:
                ang00 = 360.e0 - ang00

            eb[0, ii] = Ex
            eb[1, ii] = Ey
            eb[2, ii] = Ez
            eb[3, ii] = Bx
            eb[4, ii] = By
            eb[5, ii] = Bz

        return eb

    def AH_rout(self):

        th00 = self.th00
        ww = self.ww
        fpc = self.fpc
        bb = self.bb
        pi = self.pi
        qq = self.qq
        kb = self.kb
        me = self.me
        ev = self.ev
        ipol = self.ipol

        #  PW : w/Wce  PK : kR  ( R = Larmor Radius )
        #  FP_FC : fp/fc

        te = 1.e0*ev
        vt = np.sqrt(2.e0*kb*te/me)
        wc = qq*bb/me
        radi = vt/wc
        iflg = 0

        fpc2 = fpc**2

        th = th00
        if th == 0.e0:
            th = 0.05e0
        if th == 90.e0:
            th = 89.95e0
        th = th*pi/180.e0  # [deg]→[rad]

        pw2 = ww**2
        if pw2 == fpc2:
            iflg = 1
            return

        pka = np.sin(th)**4/(pw2-fpc2)**2
        pka = pka + 4.e0/pw2*np.cos(th)**2
        if pka < 0.e0:
            iflg = 1
            return
        pka = np.sqrt(pka)

        if ipol == -1:
            pk = 2.e0 - np.sin(th)**2/(pw2-fpc2) + pka
            if pk == 0.e0:
                iflg = 1
                return

            pk = 2.e0*fpc2/pk
            pk = pw2 - pk
            if pk < 0.e0:
                iflg = 1
                return
            pk = np.sqrt(pk)

        elif ipol == +1:
            pk = 2.e0 - np.sin(th)**2/(pw2-fpc2) - pka
            if pk == 0.e0:
                iflg = 1
                return

            pk = 2.e0*fpc2/pk
            pk = pw2 - pk
            if pk < 0.e0:
                iflg = 1
                return
            pk = np.sqrt(pk)

        else:
            iflg = 1
            print(6, "N.A.  check initial ww and ipol")
            return

        vph = ww/pk
        ref_n = pk/ww

        return pk, ref_n, vph, iflg


# %%
# model1 = model_waveform_class(th00=20)
# model2 = model_waveform_class(th00=40)

# model12_eb = model1.eb + model2.eb

# # %%
# for ll in range(len(eb)):
#     plt.plot(tt[0:60], eb[ll][0:60])
# plt.show()

# # %%
# plt.plot(tt[0:60], ang00[0:60])

# %%
# main_model_waveform()を二つ足して、伝搬角解析をしてみる
# for i in range(6):
#     plt.subplot(6,1,i+1)
#     plt.plot(model1.tt[0:60], model1.eb[i][0:60])
#     plt.plot(model2.tt[0:60], model2.eb[i][0:60])
#     plt.plot(model2.tt[0:60], model12_eb[i][0:60])

# %%
# ps.store_data()
