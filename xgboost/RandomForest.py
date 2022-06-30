# 采用随机森林模型来识别
# 参考链接:https://blog.csdn.net/qq_45067943/article/details/122715577
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


def train_model():
    """
    训练随机森林模型
    :return:
    """
    # 导入数据集
    path = "./MachineLearningCVE/mosaic.csv"
    df = pd.read_csv(path, skipinitialspace=True)
    # 输出数据预览
    print(df.head())
    print(df.columns)
    print(df['Label'].unique())

    # 处理缺失值及无穷
    df.replace((np.inf, -np.inf, np.nan), 0, inplace=True)
    # 替换标签值为 0 和 1
    data = df.replace({"Label": {'Benign': 0, 'BENIGN': 0,
                                 'DoS attacks-GoldenEye': 1, "DoS attacks-Slowloris": 1,
                                 'DDOS attack-HOIC': 1, 'DDOS attack-LOIC-UDP': 1, 'DoS attacks-SlowHTTPTest': 1,
                                 'DoS attacks-Hulk': 1, "ddos": 1,
                                 'DoS slowloris': 1, 'DoS Hulk': 1
                                 }})
    features = [f for f in data.columns if f not in ['Flow ID', 'Src IP', 'Dst IP', 'Src Port', 'Timestamp', 'Label',
                                                     'Protocol', 'Destination_Port', 'Destination Port', 'Dst Port']]
    # 自变量（该数据集的前13项）
    X = data[features]

    # 因变量（该数据集的最后1项，即第14项）
    y = data['Label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.6, random_state=1)

    # 评估回归性能
    # criterion ：
    # 回归树衡量分枝质量的指标，支持的标准有三种：
    # 1）输入"mse"使用均方误差mean squared error(MSE)，父节点和叶子节点之间的均方误差的差额将被用来作为特征选择的标准，(criterion='squared_error')
    # 这种方法通过使用叶子节点的均值来最小化L2损失
    # 2）输入“friedman_mse”使用费尔德曼均方误差，这种指标使用弗里德曼针对潜在分枝中的问题改进后的均方误差
    # 3）输入"mae"使用绝对平均误差MAE（mean absolute error），这种指标使用叶节点的中值来最小化L1损失

    # 此处使用mse
    forest = RandomForestRegressor(n_estimators=1000,
                                   criterion='squared_error',
                                   random_state=1,
                                   n_jobs=-1)
    forest.fit(X_train, y_train)

    y_train_pred = forest.predict(X_train)
    y_test_pred = forest.predict(X_test)

    print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
    print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))


if __name__ == '__main__':
    train_model()