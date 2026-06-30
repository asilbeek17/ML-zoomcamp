import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score, confusion_matrix
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from collections import Counter
from sklearn.metrics import auc, roc_curve, roc_auc_score
import random
from sklearn.model_selection import KFold
from tqdm.auto import tqdm

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

# DICTS OF VALIDATION
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
y_pred_test = model.predict_proba(X_test)[:, 1]
churn_decision = (y_pred_test >= 0.5)
# print((churn_decision == y_test).mean())

customer = dicts_test[-1]
x_small = dv.transform([customer])
coef_churn = model.predict_proba(x_small)[0, 1]
# print(coef_churn, y_test[-1])

y_val = (y_val == 'yes').astype(int)

threshholds = np.linspace(0, 1, 21)

scores = []

for t in threshholds:
    score = accuracy_score(y_val, y_pred >= t)
    scores.append(score)
    # print('%.2f %.3f' % (t, score))

actual_positive = (y_val == 1)
actual_negative = (y_val == 0)

t = 0.55
predict_positive = (y_pred >= t)
predict_negative = (y_pred < t)

tp = (predict_positive & actual_positive).sum()
tn = (predict_negative & actual_negative).sum()
fp = (predict_positive & actual_negative).sum()
fn = (predict_negative & actual_positive).sum()

cm = np.array([
    [tn, fp],
    [tn, tp]
])

# ROC CURVES
threshholds = np.linspace(0, 1, 101)
scores = []

for t in threshholds:
    actual_positive = (y_val == 1)
    actual_negative = (y_val == 0)

    predict_positive = (y_pred >= t)
    predict_negative = (y_pred < t)

    tp = (predict_positive & actual_positive).sum()
    tn = (predict_negative & actual_negative).sum()
    fp = (predict_positive & actual_negative).sum()
    fn = (predict_negative & actual_positive).sum()

    scores.append((t, tp, fp, fn, tn))

df_scores = pd.DataFrame(scores, columns=['threshold', 'tp', 'fp', 'fn', 'tn'])
df_scores['tpr'] = df_scores.tp / (df_scores.tp + df_scores.fn)
df_scores['fpr'] = df_scores.fp / (df_scores.fp + df_scores.tn)

# plt.plot(df_scores.threshold, df_scores['tpr'], label='TPR')
# plt.plot(df_scores.threshold, df_scores['fpr'], label='FPR')
# plt.legend()
# plt.show()

np.random.seed(1)
y_rand = np.random.uniform(0, 1, size=len(y_val))
randf = ((y_rand >= 0.5) == y_val).mean()

def tpr_fpr_dataframe(y_val, y_pred):
    threshholds = np.linspace(0, 1, 101)
    scores = []

    for t in threshholds:
        actual_positive = (y_val == 1)
        actual_negative = (y_val == 0)

        predict_positive = (y_pred >= t)
        predict_negative = (y_pred < t)

        tp = (predict_positive & actual_positive).sum()
        tn = (predict_negative & actual_negative).sum()
        fp = (predict_positive & actual_negative).sum()
        fn = (predict_negative & actual_positive).sum()

        scores.append((t, tp, fp, fn, tn))

    df_scores = pd.DataFrame(scores, columns=['threshold', 'tp', 'fp', 'fn', 'tn'])
    df_scores['tpr'] = df_scores.tp / (df_scores.tp + df_scores.fn)
    df_scores['fpr'] = df_scores.fp / (df_scores.fp + df_scores.tn)

    return df_scores

df_rand = tpr_fpr_dataframe(y_val, y_rand)

# plt.plot(df_rand.threshold, df_rand['tpr'], label='TPR')
# plt.plot(df_rand.threshold, df_rand['fpr'], label='FPR')
# plt.legend()
# plt.show()

sum_neg = (y_val == 0).sum()
sum_pos = (y_val == 1).sum()

y_ideal = np.repeat([0, 1], [sum_neg, sum_pos])
y_ideal_pred = np.linspace(0, 1, len(y_ideal))

y_idealf = ((y_ideal_pred >= 0.7260468417317246) == y_ideal).mean()
# print(y_idealf)

df_ideal = tpr_fpr_dataframe(y_ideal, y_ideal_pred)

# plt.plot(df_scores.threshold, df_scores['tpr'], label='TPR')
# plt.plot(df_scores.threshold, df_scores['fpr'], label='FPR')
#
# plt.plot(df_rand.threshold, df_rand['tpr'], label='TPR')
# plt.plot(df_rand.threshold, df_rand['fpr'], label='FPR')
#
# plt.plot(df_ideal.threshold, df_ideal['tpr'], label='TPR')
# plt.plot(df_ideal.threshold, df_ideal['fpr'], label='FPR')
#
# plt.legend()
# plt.show()

fpr, tpr, thresholds = roc_curve(y_val, y_pred)

pos = y_pred[y_val == 1]
neg = y_pred[y_val == 0]

rand_pos = random.randint(0, len(pos) - 1)
rand_neg = random.randint(0, len(neg) - 1)

comp = pos[rand_pos] > neg[rand_neg]
success = 0
fail = 0

for i in range(1000):
    rand_pos = random.randint(0, len(pos) - 1)
    rand_neg = random.randint(0, len(neg) - 1)

    if (pos[rand_pos] > neg[rand_neg]) == 1:
        success += 1
    else:
        fail += 1

def train(df, y_train, C=1.0):
    dicts = df[categorical + numerical].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(max_iter=10000, C=C)
    model.fit(X_train, y_train)

    return dv, model

dv, model = train(df_train, y_train, C=0.0001)

def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')
    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred

y_pred = predict(df_val, dv, model)

kfold = KFold(n_splits=10, shuffle=True, random_state=1)

n_splits = 5
resultf = []

for C in tqdm([0.0001, 0.001, 0.01, 0.1, 1, 10, 100]):
    scores = []
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

    for train_idx, val_idx in tqdm(kfold.split(df_full_train)):
        df_train = df_full_train.iloc[train_idx]
        df_val = df_full_train.iloc[val_idx]

        y_train = df_train.churn.values
        y_val = df_val.churn.values

        dv, model = train(df_train, y_train, C=C)
        y_pred = predict(df_val, dv, model)

        auc_score = roc_auc_score(y_val, y_pred)
        scores.append(auc_score)

    results = "C =", C, np.mean(scores), np.std(scores)
    resultf.append(results)
    # print("C =", C, np.mean(scores), np.std(scores))


dv, model = train(df_full_train, df_full_train.churn.values, C=1.0)
y_pred = predict(df_test, dv, model)

auc_score = roc_auc_score(y_test, y_pred)
print(auc_score)