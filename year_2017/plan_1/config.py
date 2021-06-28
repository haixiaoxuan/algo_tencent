import pandas as pd
import os


pd.set_option('display.max_columns', 100000)
pd.set_option('display.max_rows', 100000)
pd.set_option('display.width', 100000)
pd.set_option('display.max_colwidth', 500)
pd.set_option('display.precision', 2)


base_dir = "/Users/xiexiaoxuan/PycharmProjects/algo_tencent/year_2017"

path_ad = os.path.join(base_dir, "data", "ad.csv")
path_app_categories = os.path.join(base_dir, "data", "app_categories.csv")
path_position = os.path.join(base_dir, "data", "position.csv")
path_test = os.path.join(base_dir, "data", "test.csv")
path_train = os.path.join(base_dir, "data", "train.csv")
path_user = os.path.join(base_dir, "data", "user.csv")
path_user_app_actions = os.path.join(base_dir, "data", "user_app_actions.csv")
path_user_installedapps = os.path.join(base_dir, "data", "user_installedapps.csv")

path_plan1_data_dir = os.path.join(base_dir, "plan_1", "data")
