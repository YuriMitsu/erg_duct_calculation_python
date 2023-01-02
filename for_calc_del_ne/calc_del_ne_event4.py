# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
# 最小UHR　：　139.1-140.3 kHz → 誤差計算結果 : 237.5-241.7 ± 4.21-4.25 /cc
# 最大UHR　：　142.6-144.0, 145.3 kHz → 誤差計算結果 : 249.8-254.7, 259.4 ± 4.32-4.36, 4.40 /cc
# fu = np.array([139.1, 140.3, 142.6, 144.0, 145.3])
fu = np.array([144.5])
dfu = 1.2207031
# HFAの周波数幅　
# 2-159.9  kHz : 1.2207031 kHz, 159.9-280くらい kHz : 2.4414062 kHz
fc = 14.299986
dfc = 0. # 本当はBからの誤差伝搬を考えたほうが良い

# %%
me = 9.1093 * 1e-31
eps0 = 8.854 * 1e-12
e = 1.602 * 1e-19
C = 4 * np.pi**2 * me * eps0 / e**2
ne = C * (fu**2-fc**2)

# %%
ne

# %%
dne = abs(C) * 2 * np.sqrt(dfu**2*abs(fu)**2+dfc**2*abs(fc)**2)
# %%
dne
# %%
