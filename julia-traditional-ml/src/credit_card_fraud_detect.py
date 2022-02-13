import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


#----------Load Data----------
df = pd.read_csv("data/creditcard.csv")
print(f"Load dataset: {df.shape}")
print()

#----------Split Data for Training & Test----------
X = df.drop('Class', axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
print("Data split for training and test:")
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
print()

#----------Scale Data----------
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#----------Upsampling----------
sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train_scaled, y_train)

X_train_sm.shape, y_train_sm.shape
print("Upsampled training data:")
print(X_train_sm.shape)
print(y_train_sm.shape)

#----------Model Training----------
clf = XGBClassifier(
    objective='binary:logistic',
    n_estimators=1000,
    learning_rate=0.1,
)

start_time = time.time()
clf.fit(
    X_train_sm, y_train_sm,
    #early_stopping_rounds=50,
    eval_metric="auc",
    eval_set=[(X_test_scaled, y_test)],
    verbose=True
)
print(f"Elapsed time: {round(time.time() - start_time, 4)} seconds ...")
# 401.1412 seconds ...

#----------Evaluation----------
y_pred_proba = clf.predict_proba(X_test_scaled)
auc = roc_auc_score(y_test, y_pred_proba[:, 1])
aucpr = average_precision_score(y_test, y_pred_proba[:, 1])
print("ROC-AUC-Test: " + str(round(auc, 4)))
print("PR-AUC-Test: " + str(round(aucpr, 4)))
print("-----------------------------------------------------------\n")

y_pred = clf.predict(X_test_scaled)

print(f"Precision score: {round(precision_score(y_test, y_pred), 4)}")
print(f"Recall score: {round(recall_score(y_test, y_pred), 4)}")
print(f"F1 score: {round(f1_score(y_test, y_pred), 4)}")

#Elapsed time: 408.4061 seconds ...
#ROC-AUC-Test: 0.9764
#PR-AUC-Test: 0.8424
#-----------------------------------------------------------
#
#Precision score: 0.8322
#Recall score: 0.8041
#F1 score: 0.8179

