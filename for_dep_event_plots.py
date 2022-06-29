# %%
import os
import pandas as pd

# %%
data = pd.read_csv('/Users/ampuku/Documents/duct/code/python/dep_evens.csv')

# %%
# timespan, '2017-07-14/02:40:00', 20, /minute
# plot_event_normal, UHR_file_name='kuma'
# plot_kpara_ne, duct_time='2017-07-14/02:51:50', focus_f=[2., 3., 4., 5.], UHR_file_name='kuma', duct_wid_data_n=6, IorD='D' ; D


texts = '\n; コンパイル\n; .compile -v \'/Users/ampuku/Documents/duct/code/IDL/for_event_analysis/memo_dep_event_duct_time.pro\'\n\npro memo_decrease_duct_time\n'

for i in range(len(data)):
    if i == 0 or (data.start_time[i] != data.start_time[i-1] and not isinstance(data.start_time[i], float)):
        text = '\n    timespan, \'' + str(data.start_time[i]) + '\', ' + str(data.range_min[i]) + \
            ', /minute \n    plot_event_normal, UHR_file_name=\'' + \
            str(data.UHR_file_name[i]) + '\'\n'
        texts += text
    if isinstance(data.duct_time[i], str):
        text = '    plot_kpara_ne, duct_time=\'' + str(data.duct_time[i]) + '\', focus_f=' + \
            str(data.forcus_f[i]) + ', UHR_file_name=\'' + str(data.UHR_file_name[i]) + \
            '\', duct_wid_data_n=' + \
            str(data.duct_wid_data_n[i]) + ', IorD=\'' + str(data.IorD[i])
        texts += text
        if not isinstance(data.lsm[i], float):
            text = '\', lsm=' + str(data.lsm[i]) + '\n'
            texts += text
        else:
            text = '\n'
            texts += text

texts += '\n\nend'

# %%
texts

# %%
f = open('test.txt', 'w', encoding='UTF-8')
f.writelines(texts)
f.close()
# %%
os.rename('test.txt', '/Users/ampuku/Documents/duct/code/IDL/for_event_analysis/memo_dep_event_duct_time.pro')

# %%
