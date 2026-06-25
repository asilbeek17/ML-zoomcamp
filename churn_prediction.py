import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from car_prediction import df_full_train

df = pd.read_csv('Telco-Customer-Churn.csv')

df.columns = df.columns.str.lower().str.replace(' ', '_')

categorical_columns = list(df.dtypes[df.dtypes == 'str'].index)
for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')

df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
df_full_train = df_full_train.reset_index(drop=True)

y_train = df_train.churn.values
y_val = df_val.churn.values
y_test = df_test.churn.values

del df_train['churn']
del df_val['churn']
del df_test['churn']
df_full_train.churn = (df_full_train.churn == 'yes').astype('int')

global_churn_rate = df_full_train.churn.mean()
round(global_churn_rate, 2)


numerical = ['tenure', 'monthlycharges', 'totalcharges']
categorical = ['gender', 'seniorcitizen', 'partner', 'dependents',
       'phoneservice', 'multiplelines', 'internetservice',
       'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport',
       'streamingtv', 'streamingmovies', 'contract', 'paperlessbilling',
       'paymentmethod']

# churn_female = df_full_train[df_full_train['gender'] == 'female'].churn.mean()
# churn_male = df_full_train[df_full_train['gender'] == 'male'].churn.mean()
#
# churn_partner_yes = df_full_train[df_full_train['partner'] == 'yes'].churn.mean()
# churn_partner_no = df_full_train[df_full_train['partner'] == 'no'].churn.mean()

for c in categorical:
    # print(c)
    df_group = df_full_train.groupby(c).churn.agg(['mean', 'count'])
    df_group['diff'] = df_group['mean'] - global_churn_rate
    df_group['risk'] = df_group['mean'] / global_churn_rate
    # print(df_group)
    # print()

def mutual_info_churn_score(series):
    return mutual_info_score(series, df_full_train.churn)

mi = df_full_train[categorical].apply(mutual_info_churn_score).sort_values(ascending=False)
correlaction = df_full_train[numerical].corrwith(df_full_train.churn)

# DICTS OF TRAIN
train_dicts = df_train[categorical + numerical].to_dict(orient='records')
dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dicts)

# DICTS OF VAL
val_dicts = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dicts)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

a = np.linspace(-10, 10, 5)

model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)
y_pred = model.predict_proba(X_val)[:, 1]
churn_decision = y_pred >= 0.5
model_correct_percentage = ((y_val == 'yes').astype(int) == churn_decision.astype(int)).mean()

zipped = dict(zip(dv.get_feature_names_out(), model.coef_[0].round(3)))

y_test = (y_test == 'yes').astype(int)
y_full_train = df_full_train.churn.values

dicts_full_train = df_full_train[categorical + numerical].to_dict(orient='records')
dv = DictVectorizer(sparse=False)
X_full_train = dv.fit_transform(dicts_full_train)
model = LogisticRegression(max_iter=10000)
model.fit(X_full_train, y_full_train)

dicts_test = df_test[categorical + numerical].to_dict(orient='records')
X_test = dv.transform(dicts_test)
y_pred = model.predict_proba(X_test)[:, 1]
churn_decision = (y_pred >= 0.5)
# print((churn_decision == y_test).mean())

customer = dicts_test[-1]
x_small = dv.transform([customer])
coef_churn = model.predict_proba(x_small)[0, 1]
# print(coef_churn, y_test[-1])