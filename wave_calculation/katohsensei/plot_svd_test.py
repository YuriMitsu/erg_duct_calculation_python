# %%
import pytplot
import pyspedas as pys
import numpy as np
import pandas as pd
import mag_svd
import matplotlib.pyplot as plt
import importlib

# %%
'''
結果の表示にpyspedasを使いました。
お手元のpython環境にpyspedas、pytplotが入っていない場合、新たにインストールいただくか、
出力データをmatplotlibなどでplotいただくかになると思います。よろしくお願いいたします。
'''

# %%
# モジュールの再読み込み : 自作モジュール内を書き換えた時は実行
importlib.reload(mag_svd)

# %%
# test用データ読み込み
# waveformデータの保存先pathを指定
path = '/Users/ampuku/Documents/duct/code/python/wave_calculation/data/'
time = pd.read_csv(path + 'wfc_time.txt', header=None)
bx_waveform = pd.read_csv(path + 'wfc_bx_waveform.txt', header=None)
by_waveform = pd.read_csv(path + 'wfc_by_waveform.txt', header=None)
bz_waveform = pd.read_csv(path + 'wfc_bz_waveform.txt', header=None)

# %%
'''
mag_svd.mag_svd  磁場波形データを入力するとSVD解析(Santolik et al.,2003)を実行

入力データ
  time : 磁場波形データの時間
  bx_waveform : 磁場波形データ x成分
  by_waveform : 磁場波形データ y成分
  bz_waveform : 磁場波形データ z成分
SVD解析のパラメータ設定
  nfft (デフォルト　4096) : fft点数
  stride (デフォルト　2048) : fftする際に時間方向にstride点数ずつずらして実行
  n_average (デフォルト　3) : fft後にデータを時間方向に平均する際の平均点数
出力データ設定
  tplot (デフォルト　0) : 0の場合、dataname.time が時間、dataname.y が各データ値、dataname.v が周波数軸の形のclassで出力、classはmag_svd.py内 data_formで定義
                        1の場合、pyspedasのplot作成用のデータ形式 pytplotとして結果を保存

出力データ
  bspec : 磁場強度
  waveangle_th_magsvd : kベクトルの磁力線に対する角度
  waveangle_phi_magsvd : kベクトルの方位角方向の角度
  polarization_magsvd : 偏波度を表す
  planarity_magsvd : 平面性を表す

'''

# bspec, waveangle_th_magsvd, waveangle_phi_magsvd, polarization_magsvd, planarity_magsvd = mag_svd.mag_svd(time, bx_waveform, by_waveform, bz_waveform, nfft=4096, stride=2048,f_average=3, t_average=3, tplot=0)


# %%
# tplot=1　 　pytplotの形で出力
mag_svd.mag_svd(time, bx_waveform, by_waveform, bz_waveform, nfft=1024, stride=1024, f_average=3, t_average=3, tplot=1)

# %%
# pytplotで表示
pytplot.timespan('2017-07-03 04:32:00', 55,  keyword='minute')

# %%
pytplot.options(['bspec', 'waveangle_th_magsvd', 'waveangle_phi_magsvd', 'polarization_magsvd', 'planarity_magsvd'], option='ylog', value=0)
pytplot.options(['bspec', 'waveangle_th_magsvd', 'waveangle_phi_magsvd', 'polarization_magsvd', 'planarity_magsvd'], option='yrange', value=[2000, 10000])
pytplot.tplot(['bspec', 'waveangle_th_magsvd', 'waveangle_phi_magsvd', 'polarization_magsvd', 'planarity_magsvd'], save_png='test_fig')


# %%
pytplot.options(['bspec', 'waveangle_th_magsvd_mask', 'waveangle_phi_magsvd_mask', 'polarization_magsvd_mask', 'planarity_magsvd_mask'], option='ylog', value=0)
pytplot.options(['bspec', 'waveangle_th_magsvd_mask', 'waveangle_phi_magsvd_mask', 'polarization_magsvd_mask', 'planarity_magsvd_mask'], option='yrange', value=[2000, 10000])
pytplot.tplot(['bspec', 'waveangle_th_magsvd_mask', 'waveangle_phi_magsvd_mask', 'polarization_magsvd_mask', 'planarity_magsvd_mask'], save_png='test_fig')


# %%
# メモ

# pytplot.store_data('waveangle_th_magsvd', data={'x': time, 'y': bspec, 'v': freq})


