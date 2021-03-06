# -*- coding: utf-8 -*-
"""
baseline 2: ad.csv (creativeID/adID/camgaignID/advertiserID/appID/appPlatform) + lr
"""

import zipfile
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

pd.set_option('display.max_columns', None)

# 载入数据
dfTrain = pd.read_csv('data/train.csv')
dfTest = pd.read_csv('data/test.csv')
dfAd = pd.read_csv('data/ad.csv')

# 处理数据
"""merger() 函数相当于数据库的左右连接，按照creativeID 相同的数据来凝结为一行"""
dfTrain = pd.merge(dfTrain, dfAd, on='creativeID')
dfTest = pd.merge(dfTest, dfAd, on='creativeID')
y_train = dfTrain['label'].values

# feature engineering/encoding
enc = OneHotEncoder()
feats = ["creativeID", "adID", "camgaignID", "advertiserID", "appID", "appPlatform"]
for i, feat in enumerate(feats):
    x_train = enc.fit_transform(dfTrain[feat].values.reshape(-1, 1))  # (0, 2966)	1.0
    x_test = enc.transform(dfTest[feat].values.reshape(-1, 1))
    if i == 0:
        X_train, X_test = x_train, x_test
    else:
        X_train, X_test = sparse.hstack((X_train, x_train)), sparse.hstack((X_test, x_test))

# 模型训练
lr = LogisticRegression()
lr.fit(X_train, y_train)
proba_test = lr.predict_proba(X_test)[:, 1]

# submission
df = pd.DataFrame({"instanceID": dfTest["instanceID"].values, "proba": proba_test})
df.sort_values("instanceID", inplace=True)
df.to_csv('submission.csv', index=False)
with zipfile.ZipFile('submission.zip', 'w') as fout:
    fout.write('submission.csv', compress_type=zipfile.ZIP_DEFLATED)