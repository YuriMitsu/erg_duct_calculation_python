# %%
import numpy as np
import pyspedas as ps
import pytplot
import matplotlib.pyplot as plt
import pandas as pd
import streamlit
import altair as alt
from sklearn.linear_model import HuberRegressor
from sklearn.datasets import make_regression
import subprocess
import os


# %%
def calc_lsm_from_kpara(time, kpara_data, v, duct_time, duct_wid_data_n, forcus_f):

    duct_time_double = ps.time_double(duct_time)
    idx_t = np.argmin(np.abs(time - duct_time_double))

    print(idx_t)

    # ; ダクトの全幅分のkparaを持ってくる → 周波数方向を残して平均をとる

    forcus_f_sp = forcus_f[1:-1].split(',')
    print(forcus_f_sp)
    forcus_f_min = int(forcus_f_sp[0][0:-1])
    forcus_f_max = int(forcus_f_sp[-1][0:-1])

    idx_fmin = np.argmin(np.abs(v - forcus_f_min))
    idx_fmax = np.argmin(np.abs(v - forcus_f_max))

    kpara_data_idx_f = kpara_data[:, int(idx_fmin):int(idx_fmax)]
    v_idx_f = v[int(idx_fmin):int(idx_fmax)]

    kpara_data_idx_t = np.nanmean(kpara_data_idx_f[int(
        idx_t-duct_wid_data_n):int(idx_t+duct_wid_data_n), :], axis=0)
    print(np.shape(v_idx_f), np.shape(kpara_data_idx_t))
    plt.scatter(v_idx_f, kpara_data_idx_t)
    plt.show()
    n_nanind = np.where(~np.isnan(kpara_data_idx_t))

    # ; ロバスト回帰でダクト中心でのkparaを直線に当てはめる
    lam = LAM_test(v_idx_f[n_nanind[0]], kpara_data_idx_t[n_nanind[0]])

    plt.scatter(v_idx_f[n_nanind[0]], kpara_data_idx_t[n_nanind[0]])
    plt.xlim([forcus_f_min, forcus_f_max])
    plt.plot(v_idx_f[n_nanind[0]], lam[0] *
             v_idx_f[n_nanind[0]]+lam[1], color='red')
    plt.show()

    return lam


def LAM_test(x, y):
    df = pd.DataFrame({
        'x_axis': x,
        'y_axis': y
    })
    epsilon = streamlit.slider('Select epsilon',
                               min_value=1.00, max_value=10.00, step=0.01, value=1.35)

    # ロバスト回帰実行

    huber = HuberRegressor(epsilon=epsilon
                           ).fit(
        df['x_axis'].values.reshape(-1, 1),
        df['y_axis'].values.reshape(-1, 1)
    )

    # ロバスト線形回帰の係数を取得

    a1 = huber.coef_[0]
    b1 = huber.intercept_

    return [a1, b1]


def EventListCsvToProForCalcLam(file_name):
    data = pd.read_csv(
        '/Users/ampuku/Documents/duct/code/python/event_lists/'+file_name)

    texts = '\npro memo_to_save_kpara_tplots\n'

    for i in range(len(data)):
        if i == 0 or (data.start_time[i] != data.start_time[i-1] and not isinstance(data.start_time[i], float)):
            text = '\n    timespan, \'' + str(data.start_time[i]) + '\', ' + str(data.range_min[i]) + \
                ', /minute \n    save_kpara_LASVD_ma3_mask_tplots\n'
            texts += text

    texts += '\n\nend'

    actual_path = '/Users/ampuku/Documents/duct/code/python/event_list_csv_to_pro/'
    new_path = '/Users/ampuku/Documents/duct/code/IDL/tplots/kpara_LASVD_ma3_mask_tplots/'

    f = open(actual_path + 'test.txt', 'w', encoding='UTF-8')
    f.writelines(texts)
    f.close()

    if not os.path.isdir(new_path):
        os.mkdir(new_path)
    os.rename(actual_path + 'test.txt',
              new_path + 'memo_to_save_kpara_tplots.pro')


# %%
file_path = 'depression/dep_201704.csv'

# %%
data = pd.read_csv(
    '/Users/ampuku/Documents/duct/code/python/event_lists/'+file_path)

# %%
# proファイルの作成、読めなくてよし、ひとまず実行できるやつ
EventListCsvToProForCalcLam(file_path)

# %%
# ここなぜかうまくいかない.
# 作成した memo_to_save_kpara_tplots.pro を python から実行.
# 20170411080000kpara_LASVD_ma3_mask.tplot などのデータを作成する.

subprocess.run(["bash", "test.sh"])
# bash test.sh > test.log &

# p1 = subprocess.Popen(
#     ["echo", "-e", "'.r /Users/ampuku/Documents/duct/code/IDL/tplots/kpara_LASVD_ma3_mask_tplots/memo_to_save_kpara_tplots.pro'"], stdout=subprocess.PIPE)
# p2 = subprocess.Popen(["idl"], stdin=p1.stdout, stdout=subprocess.PIPE)
# p1.stdout.close()
# output = p2.communicate()[0]  # history から 'python' を含む行を検索した結果
# p2.returncode  # grep の終了コード

# この作業はコマンドプロンプトで以下を実行することにあたる.
# echo -e '.r /Users/ampuku/Documents/duct/code/IDL/tplots/kpara_LASVD_ma3_mask_tplots/memo_to_save_kpara_tplots.pro' | idl

# %%
# 以下を実行しなくていいように、自分で start_time を設定しておくことにする.（たぶん、忘れてなければ）
# for i in range(len(data)):
#     if not isinstance(data.start_time[i], float):
#         st = data.start_time[i]
#     else:
#         data.start_time[i] = st


# %%
for i in range(len(data)):
    if not isinstance(data.duct_time[i], float):
        st = data.start_time[i]
        file_name = '/Users/ampuku/Documents/duct/code/IDL/tplots/kpara_LASVD_ma3_mask_tplots/' + \
            st[0:4]+st[5:7]+st[8:10]+st[11:13]+st[14:16] + \
            st[17:19]+'kpara_LASVD_ma3_mask.tplot'
        pytplot.tplot_restore(file_name)
        pytplot.options('kpara_LASVD_ma3_mask', option='spec', value=1)
        # pytplot.tplot('kpara_LASVD_ma3_mask')
        dim = data.iloc[i]
        kpara_data = pytplot.get_data('kpara_LASVD_ma3_mask')
        lam = calc_lsm_from_kpara(kpara_data.times, kpara_data.y,
                                  kpara_data.v, dim.duct_time, dim.duct_wid_data_n, dim.forcus_f)
        data.lsm[i] = str(lam)
    else:
        data.lsm[i] = str(lam)

# %%
data.to_csv('/Users/ampuku/Documents/duct/code/python/event_lists/' +
            file_path, index=False)

# %%
