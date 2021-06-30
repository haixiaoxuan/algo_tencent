# -*- coding: utf-8 -*-
"""
baseline 1: history pCVR of creativeID/adID/camgaignID/advertiserID/appID/appPlatform
"""

import zipfile
import numpy as np
import pandas as pd

"""
pandas中关于DataFrame 去除省略号
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)
"""
pd.set_option('display.max_columns', None)

# 载入数据
dfTrain = pd.read_csv('data/train.csv')
dfTest = pd.read_csv('data/test.csv')
dfAd = pd.read_csv('data/ad.csv')

# 处理数据
"""merger() 函数相当于数据库的左右连接，按照creativeID 相同的数据来凝结为一行"""
dfTrain = pd.merge(dfTrain, dfAd, on='creativeID')
dfTest = pd.merge(dfTest, dfAd, on='creativeID')
y_train = dfTrain['label'].values  # [0 0 0 ... 0 0 0]

# 创建模型
key = 'appID'

"""
groupby(): 将数据进行排列分组，数据按照appid，
"""

dfCvr = dfTrain.groupby(key).apply(lambda df: np.mean(df['label'])).reset_index()

dfCvr.columns = [key, 'avg_cvr']
dfTest = pd.merge(dfTest, dfCvr, how='left', on=key)
dfTest['avg_cvr'].fillna(np.mean(dfTrain['label']), inplace=True)
proba_test = dfTest['avg_cvr'].values

# submission
df = pd.DataFrame({"instanceID": dfTest['instanceID'].values, "proba": proba_test})
df.sort_values("instanceID", inplace=True)
print(df.head(10))
# df.to_csv("submission.csv", index=False)
# with zipfile.ZipFile('submission.zip', 'w') as fout:
#     fout.write('submission.csv', compress_type=zipfile.ZIP_DEFLATED)