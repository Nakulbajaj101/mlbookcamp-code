#!/usr/bin/env python

# Import Libraries
import pickle

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, train_test_split
from tqdm.auto import tqdm

# Import dataset
filepath = '../chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv'
df = pd.read_csv(filepath_or_buffer=filepath)


# Config for numerical and categorical columns

categorical = ['gender', 'seniorcitizen', 'partner', 'dependents',
               'phoneservice', 'multiplelines', 'internetservice',
               'onlinesecurity', 'onlinebackup', 'deviceprotection',
               'techsupport', 'streamingtv', 'streamingmovies',
               'contract', 'paperlessbilling', 'paymentmethod']

numerical = ['tenure', 'monthlycharges', 'totalcharges']

# Config for cross validation and hyperparameters

C = [1, 0.01, 0.001, 0.0001, 0.05, 5, 10, 100]
splits = 10
seed = 42

# Data Processing

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].mean())

df.columns = df.columns.str.lower().str.replace(' ', '_')

string_columns = list(df.dtypes[df.dtypes == 'object'].index)

for col in string_columns:
    df[col] = df[col].str.lower().str.replace(' ', '_')

df.churn = (df.churn == 'yes').astype(int)


# Splitting the dataset in train, validation and testing

df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=seed)

y_train = df_train_full.churn.values
y_test = df_test.churn.values

# Dropping target variable from dataframes

del df_train_full['churn']
del df_test['churn']

# data transformation

train_dict = df_train_full[categorical + numerical].to_dict(orient='records')
test_dict = df_test[categorical + numerical].to_dict(orient='records')


dv = DictVectorizer(sparse=False)
dv.fit(train_dict)

X_train = dv.transform(train_dict)

X_test = dv.transform(test_dict)


# Model Cross Validation and hyperparameter Tuning

kfold = KFold(n_splits=splits, shuffle=True, random_state=seed)


scores = []
print("Starting cross validation")
for cost in tqdm(C):
    aucs = []
    for train_idx, val_idx in tqdm(kfold.split(X_train)):
        X = X_train[train_idx]
        X_val = X_train[val_idx]
        y = y_train[train_idx]
        y_val = y_train[val_idx]
        
        model = LogisticRegression(C=cost, max_iter=10000, random_state=seed)
        model.fit(X, y)

        y_pred = model.predict_proba(X_val)[:,1]

        auc = roc_auc_score(y_true=y_val, y_score=y_pred)
        aucs.append(auc)
    print(f"Average Auc for cost {cost} is {round(np.mean(aucs),3)} +- {round(np.std(aucs),3)}")
    scores.append((cost, round(np.mean(aucs),3), round(np.std(aucs),3)))

df_scores = pd.DataFrame(data=scores, columns=['cost', 'auc_mean', 'auc_std'])
df_scores_auc_min = df_scores[df_scores.auc_mean == df_scores.auc_mean.max()]
df_optimised = df_scores_auc_min[df_scores_auc_min.auc_std == df_scores_auc_min.auc_std.min()]
cost_optimized = df_optimised[df_optimised.cost == df_optimised.cost.min()]['cost'][0]
auc_mean_optimised = df_scores[df_scores.cost == cost_optimized]['auc_mean'][0]
auc_std_optimised = df_scores[df_scores.cost == cost_optimized]['auc_std'][0]


# Train the model on the optimized cost and test whether auc is accepted
model = LogisticRegression(C=cost_optimized, random_state=seed, max_iter=10000)
model.fit(X_train, y_train)

y_pred = model.predict_proba(X_test)[:,1]
auc_testing = roc_auc_score(y_test, y_pred)
print(f'AUC for testing dataset is {auc_testing}')


# Testing if testing is close to validation, else do further tuning
assert np.isclose(auc_testing, auc_mean_optimised, rtol=2*auc_std_optimised)

# Training final model
dv = DictVectorizer(sparse=False)

train_dict = df[numerical+categorical].to_dict(orient='records')
Y = df.churn.values
X = dv.fit_transform(train_dict)

final_model = LogisticRegression(C=cost_optimized, random_state=seed, max_iter=10000)
model.fit(X, Y)

print("Saving the model")
with open('model.pkl', 'wb') as model_file:
    pickle.dump((dv, model), model_file)
