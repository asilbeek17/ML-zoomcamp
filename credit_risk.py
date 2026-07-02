import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt

df = pd.read_csv('CreditScoring.csv')
df.columns = df.columns.str.lower().str.replace(' ', '_')

status_values = {
    1: 'ok',
    2: 'default',
    0: 'unk'
}

df.status = df.status.map(status_values)

home_values = {
    1: 'rent',
    2: 'owner',
    3: 'private',
    4: 'ignore',
    5: 'parents',
    6: 'other',
    0: 'unk',
}

df.home = df.home.map(home_values)

marital_values = {
    1: 'single',
    2: 'married',
    3: 'widow',
    4: 'separated',
    5: 'divorced',
    0: 'unk',
}

df.marital = df.marital.map(marital_values)

records_values = {
    1: 'no',
    2: 'yes',
    0: 'unk',
}

df.records = df.records.map(records_values)

job_values = {
    1: 'fixed',
    2: 'partime',
    3: 'frilance',
    4: 'others',
    0: 'unk',
}

df.job = df.job.map(job_values)

df.income = df.income.replace(to_replace=99999999, value=np.nan)
df.debt = df.debt.replace(to_replace=99999999, value=np.nan)
df.assets = df.assets.replace(to_replace=99999999, value=np.nan)

df = df[df.status != 'unk'].reset_index(drop=True)

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=11)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=11)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.status = (df_train.status == 'default').astype('int').values
y_val = df_val.status = (df_val.status == 'default').astype('int').values
y_test = df_test.status = (df_test.status == 'default').astype('int').values

del df_train['status']
del df_val['status']
del df_test['status']

train_dicts = df_train.fillna(0).to_dict(orient='records')

dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dicts)
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

val_dicts = df_val.fillna(0).to_dict(orient='records')
X_val = dv.transform(val_dicts)

dt = DecisionTreeClassifier(max_depth=1)
dt.fit(X_train, y_train)

y_pred_v = dt.predict_proba(X_val)[:, 1]
y_pred_t = dt.predict_proba(X_train)[:, 1]
auc_v = roc_auc_score(y_val, y_pred_v)
auc_t = roc_auc_score(y_train, y_pred_t)

# print('val=', auc_v)
# print('train=', auc_t)

# for d in [1, 2, 3, 4, 5, 6, 10, 15, 20, None]:
#     dt = DecisionTreeClassifier(max_depth=d)
#     dt.fit(X_train, y_train)
#
#     y_pred = dt.predict_proba(X_val)[:, 1]
#     auc = roc_auc_score(y_val, y_pred)
#
#     print('%4s -> %.3f' % (d, auc))


# for d in [4, 5, 6]:
#     for s in [2, 3, 5, 7, 10, 100, 200, 500]:
#         dt = DecisionTreeClassifier(max_depth=d, min_samples_split=s)
#         dt.fit(X_train, y_train)
#
#         y_pred = dt.predict_proba(X_val)[:, 1]
#         auc = roc_auc_score(y_val, y_pred)
#
#         print('(%4s, %3d) -> %.3f' % (d, s, auc))

dt = DecisionTreeClassifier(max_depth=6, min_samples_leaf=15)
dt.fit(X_train, y_train)

rf = RandomForestClassifier(n_estimators=10, random_state=1)
rf.fit(X_train, y_train)
y_pred = rf.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_pred)
# print(auc)

# for i in range(10, 201, 10):
#     rf = RandomForestClassifier(n_estimators=i, random_state=1)
#     rf.fit(X_train, y_train)
#     y_pred = rf.predict_proba(X_val)[:, 1]
#     auc = roc_auc_score(y_val, y_pred)
#
#     print(i, '->', auc)

# scores = []
# for d in (5, 15, 20, 30):
#     for n in range(10, 201, 10):
#         rf = RandomForestClassifier(n_estimators=n, max_depth=d, random_state=1)
#         rf.fit(X_train, y_train)
#         y_pred = rf.predict_proba(X_val)[:, 1]
#         auc = roc_auc_score(y_val, y_pred)
#
#         scores.append((d, n, auc))
#
# columns = ['max_depth', 'n_estimators', 'auc']
# df_scores = pd.DataFrame(scores, columns=columns)
#
# for d in (5, 15, 20, 30):
#     df_subset = df_scores[df_scores.max_depth == d]
#     plt.plot(df_subset.n_estimators, df_subset.auc, label=d)
#
# plt.legend(loc='best')
# plt.show()

# scores = []
# for m in [5, 15, 20, 30, 100, 200, 300]:
#     for n in range(10, 201, 10):
#         rf = RandomForestClassifier(n_estimators=n, max_depth=max_depth, min_samples_leaf=m, random_state=1)
#         rf.fit(X_train, y_train)
#         y_pred = rf.predict_proba(X_val)[:, 1]
#         auc = roc_auc_score(y_val, y_pred)
#
#         scores.append((m, n, auc))
#
# columns = ['min_samples_leaf', 'n_estimators', 'auc']
# df_scores = pd.DataFrame(scores, columns=columns)
# df_scores.head()
#
# for m in [5, 15, 20, 30, 100, 200, 300]:
#     df_subset = df_scores[df_scores.min_samples_leaf == m]
#     plt.plot(df_subset.n_estimators, df_subset.auc, label=m)
#
# plt.legend(loc='best')
# plt.show()

max_depth = 10
min_samples_leaf = 3
n=10

rf = RandomForestClassifier(n_estimators=n, max_depth=max_depth,
                            min_samples_leaf=min_samples_leaf, random_state=1)
rf.fit(X_train, y_train)

# till here for now