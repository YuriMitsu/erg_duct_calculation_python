# %%
import os
import pandas as pd
import numpy as np


def EventListCsvToPro(file_names):
    for file_name in file_names:
        data = pd.read_csv(
            '/Users/ampuku/Documents/duct/code/python/event_lists/'+file_name+'.csv')
        classification_name = file_name.split('/')[0]
        list_name = file_name.split('/')[1]

        texts = '\n; コンパイル\n; .compile -v \'/Users/ampuku/Documents/duct/code/IDL/for_event_analysis__lists/' + \
            str(classification_name) + '/memo_' + \
            str(list_name) + '_event_list.pro\''
        texts += '\n; .compile -v \'/Users/ampuku/Documents/duct/code/IDL/for_event_analysis/plot_event_normal.pro\''
        texts += '\n; .compile -v \'/Users/ampuku/Documents/duct/code/IDL/for_event_analysis/event_analysis_duct.pro\''
        texts += '\n\npro memo_' + str(list_name) + '_event_list\n'

        for i in range(len(data)):
            if i == 0 or (data.start_time[i] != data.start_time[i-1] and not isinstance(data.start_time[i], float)):
                text = '\n    timespan, \'' + str(data.start_time[i]) + '\', ' + str(data.range_min[i]) + \
                    ', /minute \n    plot_event_normal, UHR_file_name=\'' + \
                    str(data.UHR_file_name[i]) + '\'\n'
                texts += text
            if isinstance(data.duct_time[i], str):
                text = '    ;event_analysis_duct, duct_time=\'' + str(data.duct_time[i]) + '\', focus_f=' + \
                    str(data.forcus_f[i]) + ', UHR_file_name=\'' + str(data.UHR_file_name[i]) + \
                    '\', duct_wid_data_n=' + \
                    str(data.duct_wid_data_n[i]) + \
                    ', IorD=\'' + str(data.IorD[i]) + '\''
                texts += text
                if not isinstance(data.lsm[i], float):
                    text = ', lsm=' + str(data.lsm[i]) + '\n'
                    texts += text
                else:
                    text = '\n'
                    texts += text

        texts += '\n\nend'

        actual_path = '/Users/ampuku/Documents/duct/code/python/event_list_csv_to_pro/'
        new_path = '/Users/ampuku/Documents/duct/code/IDL/for_event_analysis__lists/' + \
            str(classification_name) + '/'

        f = open(actual_path + 'test.txt', 'w', encoding='UTF-8')
        f.writelines(texts)
        f.close()

        if not os.path.isdir(new_path):
            os.mkdir(new_path)
        os.rename(actual_path + 'test.txt',
                  new_path + 'memo_' + str(list_name) + '_event_list.pro')


# %%
"""
ここにファイル名を入れて回すと一つずつmemo~.proに変換される
memo~.proを回せばイベントの時間帯の様子が細かく(多分)見れるplotが生成される(はず)
"""
file_names = np.array(['suspicion/sus_201704'])
EventListCsvToPro(file_names)

# %%
