import os
import warnings

import numpy as np
# x-列最小值）/ （列最大值-列最小值）， value在0-1, MinMaxScaler().fit_transform(iris.data)
import pandas as pd
import scipy as sp

# 特征二值化 - 大于阀值转为1，小于等于阀值为0
pd.set_option('display.max_columns', None)

# 随机森林建模&&特征重要度排序

# 随机森林调参
from sklearn.model_selection import GridSearchCV
# Xgboost调参
import xgboost as xgb
# 我们可以看到正负样本数量相差非常大，数据严重unbalanced
from blagging import BlaggingClassifier

""" 文件读取 """


def read_csv_file(filname, logging=False):
    data = pd.read_csv(filname)
    if logging:
        print(data.head(5))
        print(data.columns.values)
        print(data.describe())
        print(data.info())
    return data


""" 数据的处理 """


# 第一类编码
def categories_process_first_class(cate):
    cate = str(cate)
    if len(cate) == 1:
        if int(cate) == 0:
            return 0
    else:
        return int(cate[0])


# 第二类编码
def categories_process_second_class(cate):
    cate = str(cate)
    if len(cate) < 3:
        return 0
    else:
        return int(cate[1:])


# 年龄切断处理


def age_process(age):
    age = int(age)
    if age == 0:
        return 0
    if age < 15:
        return 1
    elif age < 25:
        return 2
    elif age < 40:
        return 3
    elif age < 60:
        return 4
    else:
        return 5


# 省份数据处理


def process_province(hometown):
    hometown = str(hometown)
    province = int(hometown[0:2])
    return province


# 城市


def process_city(hometown):
    hometown = str(hometown)
    if len(hometown) > 1:
        province = int(hometown[2:])
    else:
        province = 0
    return province


# 几点钟


def get_time_day(t):
    t = str(t)
    t = int(t[0:2])
    return t


# 一天切分成4段


def get_time_hour(t):
    t = str(t)
    t = int(t[2:4])
    if t < 6:
        return 0
    if t < 12:
        return 1
    elif t < 18:
        return 2
    else:
        return 3


# 评估与计算logloss


def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1 - epsilon, pred)
    ll = sum(act * sp.log(pred) + sp.subtract(1, act) * sp.log(sp.subtract(1, pred)))
    ll = ll * -1.0 / len(act)
    return ll


train_data = read_csv_file('data/train.csv', logging=False)  # 读取train_data和ad
ad = read_csv_file('data/ad.csv', logging=False)

app_categories = read_csv_file('data/app_categories.csv', logging=False)
app_categories['app_categories_first_class'] = app_categories['appCategory'].apply(categories_process_first_class)
# ['appID' 'appCategory' 'app_categories_first_class' 'app_categories_second_class']
app_categories['app_categories_second_class'] = app_categories['appCategory'].apply(categories_process_second_class)
user = read_csv_file('data/user.csv', logging=False)
""" 画年龄分布柱状图
age_obj = user.age.value_counts()
plt.bar(age_obj.index, age_obj.values.tolist())
plt.show()
"""
# 用户信息处理
user['age_process'] = user['age'].apply(age_process)
user['hometown_province'] = user['hometown'].apply(process_province)
user['hometown_city'] = user['hometown'].apply(process_city)
user["residence_province"] = user['residence'].apply(process_province)
user["residence_city"] = user['residence'].apply(process_city)

train_data['clickTime_day'] = train_data['clickTime'].apply(get_time_day)
train_data['clickTime_hour'] = train_data['clickTime'].apply(get_time_hour)

# test_data
test_data = read_csv_file('./data/test.csv', False)
test_data['clickTime_day'] = test_data['clickTime'].apply(get_time_day)
test_data['clickTime_hour'] = test_data['clickTime'].apply(get_time_hour)

# 全部合并
train_user = pd.merge(train_data, user, on='userID')
train_user_ad = pd.merge(train_user, ad, on='creativeID')
train_user_ad_app = pd.merge(train_user_ad, app_categories, on='appID')

# 取出数据和 label
x_user_ad_app = train_user_ad_app.loc[:, ['creativeID', 'userID', 'positionID',
                                          'connectionType', 'telecomsOperator', 'clickTime_day', 'clickTime_hour',
                                          'age', 'gender', 'education',
                                          'marriageStatus', 'haveBaby', 'residence', 'age_process',
                                          'hometown_province', 'hometown_city', 'residence_province', 'residence_city',
                                          'adID', 'camgaignID', 'advertiserID', 'appID', 'appPlatform',
                                          'app_categories_first_class', 'app_categories_second_class']]

x_user_ad_app = x_user_ad_app.values
print(x_user_ad_app)
x_user_ad_app = np.array(x_user_ad_app, dtype='int32')
print(x_user_ad_app)
# 标签部分
y_user_ad_app = train_user_ad_app.loc[:, ['label']].values

"""
# 随机森林建模&&特征重要度排序 用RF 计算特征重要度
feat_labels = np.array(['creativeID','userID','positionID',
 'connectionType','telecomsOperator','clickTime_day','clickTime_hour','age', 'gender' ,'education',
 'marriageStatus' ,'haveBaby' , 'residence' ,'age_process',
 'hometown_province', 'hometown_city','residence_province', 'residence_city',
 'adID', 'camgaignID', 'advertiserID', 'appID' ,'appPlatform' ,
 'app_categories_first_class' ,'app_categories_second_class'])
# 随机森林
forest = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
forest.fit(x_user_ad_app, y_user_ad_app.reshape(y_user_ad_app.shape[0],))
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(x_user_ad_app.shape[1]):
    print("%2d) %-*s %f" % (f+1, 30, feat_labels[indices[f]], importances[indices[f]]))
plt.title('Feature Importances')
plt.bar(range(x_user_ad_app.shape[1]), importances[indices], color='lightblue', align='center')
plt.xticks(range(x_user_ad_app.shape[1]), feat_labels[indices], rotation=90)
plt.xlim(-1, x_user_ad_app.shape[1])
plt.tight_layout()
plt.savefig('random_forest.png', dpi=300)
plt.show()
"""

"""
随机森林调参
param_grid = {
    'n_estimators': [10, 100, 500, 1000],
    'max_features': [0.6, 0.7, 0.8, 0.9]
}
rf = RandomForestClassifier()
rfc = GridSearchCV(rf, param_grid, scoring='neg_log_loss', cv=3, n_jobs=2)
rfc.fit(x_user_ad_app, y_user_ad_app.shape(y_user_ad_app[0],))
print(rfc.best_score_)
print(rfc.best_params_)
"""

"""
Xgboost 调参
"""
os.environ['OMP_NUM_THREADS'] = '8'  # 开启并行训练
rng = np.random.RandomState(4315)
warnings.filterwarnings('ignore')

param_grid = {
    "max_depth": [3, 4, 5, 7, 9],
    "n_estimators": [10, 50, 100, 400, 800, 1000, 1200],
    "learning_rate": [0.1, 0.2, 0.3],
    "gamma": [0, 0.2],
    "subsample": [0.8, 1],
    "colsample_bylevel": [0.8, 1]
}

xgb_model = xgb.XGBClassifier()
rgs = GridSearchCV(xgb_model, param_grid, n_jobs=-1)
rgs.fit(x_user_ad_app, y_user_ad_app)  # fit(x,y)
print(rgs.best_score_)
print(rgs.best_params_)

# 正负样本比
positive_num = train_user_ad_app[train_user_ad_app['label'] == 1].values.shape[0]
negative_num = train_user_ad_app[train_user_ad_app['label'] == 0].values.shape[0]

print(negative_num / float(positive_num))

"""
我们用Bagging修正过后，处理不均衡样本的B(l)agging来进行训练和实验
------------后面的这段程序就开始看不懂了--------------
"""

# 处理unbalanced 的classifier
classifier = BlaggingClassifier(n_jobs=-1)
classifier.fit(x_user_ad_app, y_user_ad_app)
classifier.predict_proba(x_test_clean)
# 预测
test_data = pd.merge(test_data, user, on='userID')
test_user_ad = pd.merge(test_data, ad, on='creativeID')
test_user_ad_app = pd.merge(test_user_ad, app_categories, on='appID')

x_test_clean = test_user_ad_app.loc[:, ['creativeID', 'userID', 'positionID',
                                        'connectionType', 'telecomsOperator', 'clickTime_day', 'clickTime_hour', 'age',
                                        'gender', 'education',
                                        'marriageStatus', 'haveBaby', 'residence', 'age_process',
                                        'hometown_province', 'hometown_city', 'residence_province', 'residence_city',
                                        'adID', 'camgaignID', 'advertiserID', 'appID', 'appPlatform',
                                        'app_categories_first_class', 'app_categories_second_class']].values

x_test_clean = np.array(x_test_clean, dtype='int32')

result_predict_prob = []

result_predict = []
for i in range(scale):
    result_indiv = clfs[i].predict(x_test_clean)
    result_indiv_proba = clfs[i].predict_proba(x_test_clean)[:, 1]
    result_predict.append(result_indiv_proba)

result_indiv_proba = np.reshape(result_predict_prob, [-1, scale])
result_predict = np.reshape(result_predict, [-1, scale])

result_predict_prob = np.mean(result_predict_prob, axis=1)
result_predict = max_count(result_predict)

result_predict_prob = np.array(result_predict_prob).reshape([-1, 1])

test_data['prob'] = result_predict_prob
test_data = test_data.loc[:, ['instanceID', 'prob']]
test_data.to_csv('predict.csv', index=False)
print
"prediction done!"
