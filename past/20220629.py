# %%
import os
import pandas as pd

# %%
data = pd.read_csv('/Users/ampuku/Documents/duct/code/python/dep_evens.csv')

# %%
# ./fufp3-txt YYYYMMDD > txt/YYYY/erg_hfa_l3_high_YYYYMMDD.txt
texts = ''

for i in range(len(data)):
    st = str(data.start_time[i])
    texts += './fufp3-txt ' + st[0:4] + st[5:7] + st[8:10] + ' > txt/' + \
        st[0:4] + '/erg_hfa_l3_high_' + st[0:4] + \
        st[5:7] + st[8:10] + '.txt\n'
# %%
texts

# %%
f = open('test.txt', 'w', encoding='UTF-8')
f.writelines(texts)
f.close()
# %%
os.rename('test.txt', '/Users/ampuku/Documents/duct/code/python/20220629test.txt')

# %%
