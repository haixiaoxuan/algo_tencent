import pandas as pd
# import modin.pandas as pd
import numpy as np
from memory_profiler import profile
from year_2017.plan_1 import config
import util
import os


n_rows = None


# 69s
@util.timer
def concat_feature():
    """ join ad,user,position,app_category """
    df_ad = pd.read_csv(config.path_ad)
    df_user = pd.read_csv(config.path_user, dtype={"userID": np.int32, "age": np.int8, "gender": np.int8, "education": np.int8,
                                                   "marriageStatus": np.int8, "haveBaby": np.int8, "hometown": np.int16, "residence": np.int16},
                          nrows=n_rows)
    df_position = pd.read_csv(config.path_position)
    df_app_category = pd.read_csv(config.path_app_categories)
    df_train = pd.read_csv(config.path_train, dtype={"label": np.int8, "clickTime": np.int32, "conversionTime": np.float,
                                                     "creativeID": np.int32, "userID": np.int32, "positionID": np.int32,
                                                     "connectionType": np.int8, "telecomsOperator": np.int8},
                           nrows=n_rows)
    # df_train = pd.read_parquet(os.path.join(config.path_data_dir, "train.parquet"))

    # join
    tmp_1 = pd.merge(df_train, df_ad, how="left", on="creativeID")
    tmp_2 = pd.merge(tmp_1, df_user, how="left", on="userID")
    tmp_3 = pd.merge(tmp_2, df_position, how="left", on="positionID")
    res = pd.merge(tmp_3, df_app_category, how="left", on="appID")

    # label  clickTime  conversionTime  creativeID    userID  positionID  connectionType  telecomsOperator
    # adID  camgaignID  advertiserID  appID  appPlatform
    # age  gender  education  marriageStatus  haveBaby  hometown  residence
    # sitesetID  positionType
    # appCategory
    res.astype(
        {"label": np.int8, "clickTime": np.int32, "creativeID": np.int32, "userID": np.int32, "positionID": np.int32,
         "connectionType": np.int8, "telecomsOperator": np.int8,
         "adID": np.int32, "camgaignID": np.int16, "advertiserID": np.int16, "appID": np.int16, "appPlatform": np.int8,
         "age": np.int8, "gender": np.int8, "education": np.int8, "marriageStatus": np.int8, "haveBaby": np.int8,
         "hometown": np.int16, "residence": np.int16,
         "sitesetID": np.int8, "positionType": np.int8, "appCategory": np.int16
         })
    res.to_parquet(os.path.join(config.path_plan1_data_dir, "concat_feature_train.parquet"))


# 85s
@util.timer
def add_time_feature():
    df = pd.read_parquet(os.path.join(config.path_plan1_data_dir, "concat_feature_train.parquet"))
    df["clickTime_day"] = df["clickTime"].apply(lambda x: str(x)[:2])
    df["clickTime_hour"] = df["clickTime"].apply(lambda x: str(x)[2:4])
    df["clickTime_min"] = df["clickTime"].apply(lambda x: str(x)[4:6])
    df["clickTime_second"] = df["clickTime"].apply(lambda x: str(x)[6:])
    print(df.head())
    df.to_parquet(os.path.join(config.path_plan1_data_dir, "add_time_feature_train.parquet"))


def add_cross_feature():
    df = pd.read_parquet(os.path.join(config.path_plan1_data_dir, "add_time_feature_train.parquet"))
    df["advertiserID_camgaignID"] = df["advertiserID"] * df["camgaignID"]
    df["positionID_positionType"] = df["positionID"] / 1000 + df["positionType"]



if __name__ == "__main__":
    # concat_feature()
    # add_time_feature()
    add_time_feature()