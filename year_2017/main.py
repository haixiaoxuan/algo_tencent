import pandas as pd


pd.set_option('display.max_columns', 100000)
pd.set_option('display.max_rows', 100000)
pd.set_option('display.width', 100000)
pd.set_option('display.max_colwidth', 500)
pd.set_option('display.precision', 2)


path_ad = r"C:\Users\xiexiaoxuan\Desktop\腾讯广告算法\2017腾讯赛数据集\final\ad.csv"
path_app_categories = r"C:\Users\xiexiaoxuan\Desktop\腾讯广告算法\2017腾讯赛数据集\final\app_categories.csv"
path_position = r"C:\Users\xiexiaoxuan\Desktop\腾讯广告算法\2017腾讯赛数据集\final\position.csv"
path_test = r"C:\Users\xiexiaoxuan\Desktop\腾讯广告算法\2017腾讯赛数据集\final\test.csv"
path_train = r"C:\Users\xiexiaoxuan\Desktop\腾讯广告算法\2017腾讯赛数据集\final\train.csv"
path_user = r"C:\Users\xiexiaoxuan\Desktop\腾讯广告算法\2017腾讯赛数据集\final\user.csv"
path_user_app_actions = r"C:\Users\xiexiaoxuan\Desktop\腾讯广告算法\2017腾讯赛数据集\final\user_app_actions.csv"
path_user_installedapps = r"C:\Users\xiexiaoxuan\Desktop\腾讯广告算法\2017腾讯赛数据集\final\user_installedapps.csv"



if __name__ == '__main__':
    df = pd.read_csv(path_ad)
    print(df.info())


