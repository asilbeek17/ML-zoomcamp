from os import pread

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
# graph = sns.histplot(price_logs, bins=50)
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

# y original msrp price
y11 = int(df_train['msrp'].iloc[1])
y21 = int(df_train['msrp'].iloc[2])
y31 = int(df_train['msrp'].iloc[3])
y41 = int(df_train['msrp'].iloc[4])
y51 = int(df_train['msrp'].iloc[5])
y = [y11, y21, y31, y41, y51]

del df_train['msrp']
del df_test['msrp']
del df_val['msrp']

xi = [275, 13, 1385]
x11 = int(df_train['engine_hp'].iloc[1]); x12 = int(df_train['city_mpg'].iloc[1])\
    ; x13 = int(df_train['popularity'].iloc[1])
x21 = int(df_train['engine_hp'].iloc[2]); x22 = int(df_train['city_mpg'].iloc[2])\
    ; x23 = int(df_train['popularity'].iloc[2])
x31 = int(df_train['engine_hp'].iloc[3]); x32 = int(df_train['city_mpg'].iloc[3])\
    ; x33 = int(df_train['popularity'].iloc[3])
x41 = int(df_train['engine_hp'].iloc[4]); x42 = int(df_train['city_mpg'].iloc[4])\
    ; x43 = int(df_train['popularity'].iloc[4])
x51 = int(df_train['engine_hp'].iloc[5]); x52 = int(df_train['city_mpg'].iloc[5])\
    ; x53 = int(df_train['popularity'].iloc[5])

x1 = [1, x11, x12, x13]
x2 = [1, x21, x22, x23]
x3 = [1, x31, x32, x33]
x4 = [1, x41, x42, x43]
x5 = [1, x51, x52, x53]

xf = [x1, x2, x3, x4, x5]
xf = np.array(xf)
# print(xf)

w0 = 7.17
w = [0.01, 0.04, 0.002]
w_new = [w0] + w

def dot(xi, w):
    n = len(xi)
    res = 0.0

    for i in range(n):
        res = res + xi[i] * w[i]

    return res

def linear_regression(xi):
    # pred = w0 + dot(xi, w)
    # print(pred)
    pass

def linear_regression_new(xi):
    # print(xf.dot(w_new))
    pass

linear_regression_new(xi)

def train_linear_regression(xf, y):
    ones = np.ones(xf.shape[0])
    xf = np.column_stack([ones, xf])

    xtx = xf.T.dot(xf)
    inverse = np.linalg.inv(xtx)
    w_full = inverse.dot(xf.T).dot(y)

    return w_full[0], w_full[1:]

    # print(w_full[0], w_full[1:])

# train_linear_regression(xf, y)

base = ['engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg', 'popularity']
x_train = df_train[base].values
x_train = df_train[base].fillna(0).values
w0, w = train_linear_regression(x_train, y_train)
y_pred = w0 + x_train.dot(w)

# sns.histplot(y_pred, color='red', alpha=0.5, bins=50)
# sns.histplot(y_train, color='blue', alpha=0.5, bins=50)
# plt.show()

def rmse(y, y_pred):
    se = (y - y_pred) ** 2
    mse = se.mean()
    return np.sqrt(mse)

# print(rmse(y_train, y_pred))

def prepare_X(df):
    df_num = df[base]
    df_num = df_num.fillna(0)
    X = df_num.values
    return X

x_train = prepare_X(df_train)
w0, w = train_linear_regression(x_train, y_train)

x_val = prepare_X(df_val)
y_pred = w0 + x_val.dot(w)
# print(rmse(y_val, y_pred))

def prepare_X(df):
    df = df.copy()

    df['age'] = 2017 - df.year
    features = base + ['age']

    df_num = df[features]
    df_num = df_num.fillna(0)
    X = df_num.values
    return X

x_train = prepare_X(df_train)
x_train = prepare_X(df_train)
w0, w = train_linear_regression(x_train, y_train)

x_val = prepare_X(df_val)
y_pred = w0 + x_val.dot(w)
# print(rmse(y_val, y_pred))

# sns.histplot(y_pred, color='red', alpha=0.5, bins=50)
# sns.histplot(y_val, color='blue', alpha=0.5, bins=50)
# plt.show()