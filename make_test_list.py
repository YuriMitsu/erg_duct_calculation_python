# %%
import numpy as np
import pyspedas as ps
import pandas as pd


# %%
data = pd.read_csv('/Users/ampuku/Documents/duct/code/python/events.csv')

# %%
# pro ファイルの作成、読めなくてよし、ひとまず実行できるやつ
# '/Users/ampuku/Documents/duct/code/IDL/tplots/kpara_LASVD_ma3_mask_tplots/test_list.pro'

texts = 'pro test_list\n'

for i in range(len(data)):
    if i == 0 or (data.start_time[i] != data.start_time[i-1] and not isinstance(data.start_time[i], float)):
        text = '\n    timespan, \'' + str(data.start_time[i]) + '\', ' + str(data.range_min[i]) + \
            ', /minute \n    test_20220711\n'
        texts += text

texts += '\nend'
# texts
# %%

f = open('/Users/ampuku/Documents/duct/code/python/test_20220711.txt','w', encoding='UTF-8')
f.writelines(texts)
f.close()
os.rename('/Users/ampuku/Documents/duct/code/python/test_20220711.txt','/Users/ampuku/Documents/duct/code/IDL/tplots/kpara_LASVD_ma3_mask_tplots/test_list.pro')


# %%
# idlで実行してtplot変数を作成する！ここ時間がかかる..中でSVD法が動いている..
# idl
# erg_init
# .compile -v '/Users/ampuku/Documents/duct/code/IDL/tplots/kpara_LASVD_ma3_mask_tplots/test_20220711.pro'
# .compile -v '/Users/ampuku/Documents/duct/code/IDL/tplots/kpara_LASVD_ma3_mask_tplots/test_list.pro'
# test_list
