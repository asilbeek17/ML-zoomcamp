import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data.csv')
df.columns = df.columns.str.lower().str.replace(' ', '_')

types = (df.dtypes == 'str')
strings = list(df.dtypes[types].index)

for i in strings:
    df[i] = df[i].str.lower().str.replace(' ', '_')

np.log1p([1, 10, 100, 1000, 10000])
price_logs = np.log1p(df.msrp)
graph = sns.histplot(price_logs, bins=50)
# plt.show()

n = len(df)

n_val = int(n * 0.2)
n_test = int(n * 0.2)
n_train = n - n_val - n_test

np.random.seed(2)
idx = np.arange(n)
np.random.shuffle(idx)

df_val = df.iloc[idx[:n_val]]
df_test = df.iloc[idx[n_val:n_val+n_test]]
df_train = df.iloc[idx[n_val+n_test:]]

df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)

y_train = np.log1p(df_train.msrp.values)
y_test = np.log1p(df_test.msrp.values)
y_val = np.log1p(df_val.msrp.values)
del df_train['msrp']
del df_test['msrp']
del df_val['msrp']

print(y_train)
