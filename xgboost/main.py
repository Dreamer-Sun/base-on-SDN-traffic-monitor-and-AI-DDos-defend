"""
智能DDos检测防御
基于xgboost-lr模型 和 lightgbm模型      来检测
"""
# -*- coding: utf-8 -*-
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import warnings
import xgboost as xgb


def cv_model(clf, x_train, y_train, clf_name):
    '''
        @param clf: 分类器包名(lgb&xgb)、分类器实例名(CatBoostClassifier)
        @param x_train: 训练集X
        @param y_train: 训练集y
        @param clf_name: 分类器名：'lgb' 'xgb' 'cat'

        @return: model 最后一次交叉训练后的模型
        @return: train 分类器在训练集上的预测结果
    '''

    folds = 5
    seed = 2020
    # use StratifiedKFold instead of KFold to solve unbalanced datasets
    # StratifiedKFold 将X_train和 X_test 做有放回抽样，随机分三次，取出索引 交叉验证

    kf = StratifiedKFold(n_splits=folds,        # 将数据划分为 fold份
                         shuffle=True,          # 打乱顺序
                         random_state=seed)     # 随机数的种子

    train = np.zeros(x_train.shape[0])  # 返回一个0填充数组

    cv_accuracy_scores = []
    cv_f1_scores = []
    cv_auc_scores = []

    for i, (train_index, valid_index) in enumerate(kf.split(x_train, y_train)):
        print('************************************ {0}/{1} ************************************'.format(str(i + 1),
                                                                                                         folds))
        trn_x, trn_y, val_x, val_y = x_train.iloc[train_index], y_train[train_index], x_train.iloc[valid_index], \
                                     y_train[valid_index]

        if clf_name == "lgb":
            train_matrix = clf.Dataset(trn_x, label=trn_y)
            valid_matrix = clf.Dataset(val_x, label=val_y)

            params = {
                'boosting_type': 'gbdt',    # boosting_type (string, optional (default=‘gbdt’))
                                            # ‘gbdt’, traditional Gradient Boosting Decision Tree.  梯度提升决策树
                                            # ‘dart’, Dropouts meet Multiple Additive Regression Trees. 使用单边梯度抽样算法，速度很快，但是可能欠拟合。
                                            # ‘goss’, Gradient-based One-Side Sampling. GOSS是一个样本实例的采样算法，目的是丢弃一些对计算信息增益没有帮助的实例留下有帮助的
                                            # ‘rf’, Random Forest 使用随机森林
                'objective': 'binary',      # Specify the learning task and the corresponding learning objective or a custom objective function to be used (see note below). Default: ‘regression’ for LGBMRegressor, ‘binary’ or ‘multiclass’ for LGBMClassifier, ‘lambdarank’ for LGBMRanker.
                                            # 指定要使用的学习任务和相应的学习目标或自定义目标函数（请参见下面的注释）。默认值：LGBMRegressor为“回归”，LGBMClassifier为“二分类”或“多类”，LGBMRanker为“lambdarank”。
                                            # 此处为二分类
                'metric': 'auc',            # 用于指定评估指标 ‘auc’，用于二分类任务。
                'min_child_weight': 5,      # 叶节点样本的最少数量，默认值20，用于防止过拟合
                'num_leaves': 2 ** 5,       # 指定叶子的个数，默认值为31，此参数的数值应该小于 2^{max_depth}2max_depth。
                'lambda_l2': 10,            # 此参数服务于L2正则化，一般也是在0-1000的范围去进行调参。如果有非常强势的特征，可以人为加大一些reg_lambda使得整体特征效果平均一些，一般会比reg_alpha的数值略大一些，但如果这个参数大的夸张也需要再查看一遍特征是否合理。
                'feature_fraction': 0.8,    # 构建弱学习器时，对特征随机采样的比例，默认值为1。 推荐的候选值为：[0.6, 0.7, 0.8, 0.9, 1]
                'bagging_fraction': 0.8,    # 默认值1，指定采样出 subsample * n_samples 个样本用于训练弱学习器。注意这里的子采样和随机森林不一样，随机森林使用的是放回抽样，而这里是不放回抽样。 取值在(0, 1)之间，设置为1表示使用所有数据训练弱学习器。如果取值小于1，则只有一部分样本会去做GBDT的决策树拟合。选择小于1的比例可以减少方差，即防止过拟合，但是会增加样本拟合的偏差，因此取值不能太低
                'bagging_freq': 4,          # 数值型，默认值0，表示禁用样本采样。如果设置为整数 z ，则每迭代 k 次执行一次采样
                'learning_rate': 0.1,       # LightGBM 不完全信任每个弱学习器学到的残差值，为此需要给每个弱学习器拟合的残差值都乘上取值范围在(0, 1] 的 eta，设置较小的 eta 就可以多学习几个弱学习器来弥补不足的残差。 推荐的候选值为：[0.01, 0.015, 0.025, 0.05, 0.1]
                'seed': 2020,               # 随机数种子
                'nthread': 28,              # LightGBM 的线程数
                'n_jobs': 24,               # Number of parallel threads. 平行线程数
                'silent': True,             # 运行boosting时是否打印消息
                'verbose': -1,              # 可以是bool类型，也可以是整数类型。如果设置为整数，则每间隔verbose_eval次迭代就输出一次信息。
                'is_unbalance': True,       # unblanced datasets
            }

            model = clf.train(params, train_matrix, 50000, valid_sets=[train_matrix, valid_matrix],
                              verbose_eval=200, early_stopping_rounds=200)
            val_score = model.predict(val_x, num_iteration=model.best_iteration)
            # 输出前80特征重要性从高到低
            print(list(
                sorted(zip(model.feature_name(), model.feature_importance("gain")), key=lambda x: x[1], reverse=True))[
                  :80])

        if clf_name == "xgb":
            # print(trn_x, trn_y, val_x, val_y)
            # print(trn_x.isnull().any())
            # print(trn_x.isin([np.nan, np.inf, -np.inf])).any(1).sum()
            train_matrix = clf.DMatrix(trn_x, label=trn_y)
            valid_matrix = clf.DMatrix(val_x, label=val_y)

            params = {'booster': 'gbtree',
                      'objective': 'binary:logistic',
                      'eval_metric': 'auc',
                      'gamma': 1,   # 用于限制在叶子节点上进一步分裂所需的最小损失函数下降值。分裂节点时，损失函数减小值只有大于等于gamma节点才分裂。gamma越大，算法越保守。取值范围为[0,]。
                      'min_child_weight': 1.5,
                      'max_depth': 5,
                      'lambda': 10,
                      'subsample': 0.7,
                      'colsample_bytree': 0.7,
                      'colsample_bylevel': 0.7,
                      'eta': 0.04,
                      'tree_method': 'exact',
                      'seed': 2020,
                      'nthread': 36,
                      "silent": True,
                      'is_unbalance': True,  # unblanced datasets
                      }

            watchlist = [(train_matrix, 'train'), (valid_matrix, 'eval')]

            model = clf.train(params, train_matrix, num_boost_round=50000, evals=watchlist, verbose_eval=200,
                              early_stopping_rounds=200)
            val_score = model.predict(valid_matrix, ntree_limit=model.best_ntree_limit)
            # 输出前80特征重要性从高到低
            # print(list(
            #     sorted(zip(model.feature_name(), model.feature_importance("gain")), key=lambda x: x[1], reverse=True))[
            #       :80])
        # 交叉验证每次得到一部分结果
        val_pred = np.int8(np.round(val_score))
        train[valid_index] = val_pred

        cv_accuracy_scores.append(accuracy_score(val_y, val_pred))
        cv_f1_scores.append(f1_score(val_y, val_pred))
        cv_auc_scores.append(roc_auc_score(val_y, val_score))

        print(cv_accuracy_scores)
        print(cv_f1_scores)
        print(cv_auc_scores)

    print("%s_accuracy_score_list:" % clf_name, cv_accuracy_scores)
    print("%s_accuracy_score_mean:" % clf_name, np.mean(cv_accuracy_scores))
    print("%s_accuracy_score_std:" % clf_name, np.std(cv_accuracy_scores))

    print("%s_f1_score_list:" % clf_name, cv_f1_scores)
    print("%s_f1_score_mean:" % clf_name, np.mean(cv_f1_scores))
    print("%s_f1_score_std:" % clf_name, np.std(cv_f1_scores))

    print("%s_auc_score_list:" % clf_name, cv_auc_scores)
    print("%s_auc_score_mean:" % clf_name, np.mean(cv_auc_scores))
    print("%s_auc_score_std:" % clf_name, np.std(cv_auc_scores))

    return model, train


def lgb_model(x_train, y_train):
    lgb_trained_model, lgb_train = cv_model(lgb, x_train, y_train, "lgb")
    # 保存模型
    lgb_trained_model.save_model("model_lgb.txt")
    return lgb_trained_model, lgb_train


def xgb_model(x_train, y_train):
    xgb_trained_model, xgb_train = cv_model(xgb, x_train, y_train, "xgb")
    # 保存模型
    xgb_trained_model.save_model("model_xgb.txt")
    return xgb_trained_model, xgb_train


def lgb_pre(X):
    # 模型加载
    lgb_trained_model = lgb.Booster(model_file='model_lgb.txt')

    # 模型预测
    y_pred_DDos = lgb_trained_model.predict(X, num_iteration=lgb_trained_model.best_iteration)
    temp = np.ones(y_pred_DDos.shape[0])  # 返回一个1填充数组
    y_pred_Benign = temp - y_pred_DDos    # 获取每一个ddos的概率
    # 返回预测结果
    y_pred = []
    for p in range(0, len(y_pred_DDos)):
        if y_pred_DDos[p] - y_pred_Benign[p] > 0:
            y_pred.append("DDos")
        else:
            y_pred.append("Benign")
    return y_pred


def lgb_pre_label(X):
    # 模型加载
    lgb_trained_model = lgb.Booster(model_file='model_lgb.txt')

    # 模型预测
    y_pred_DDos = lgb_trained_model.predict(X, num_iteration=lgb_trained_model.best_iteration)
    temp = np.ones(y_pred_DDos.shape[0])  # 返回一个1填充数组
    y_pred_Benign = temp - y_pred_DDos  # 获取每一个ddos的概率
    # 返回预测结果
    y_pred = []
    for p in range(0, len(y_pred_DDos)):
        if y_pred_DDos[p] - y_pred_Benign[p] > 0:
            y_pred.append(1)
        else:
            y_pred.append(0)
    return y_pred


def OutputCsv(data, y_pred):
    # 测试用
    # 将输出值输出为表格
    # 新添加一列
    data['pred'] = y_pred
    print(data)
    # 输出csv
    data.to_csv('DataFrame_ab.csv', index=False, sep=',')


def OutputCsv2(path, outputname):
    # 使用另一个测试集做数据的测试
    # path = ''
    df = pd.read_csv(path, skipinitialspace=True)
    # 处理缺失值及无穷
    df.replace((np.inf, -np.inf, np.nan), 0, inplace=True)
    # 替换标签值为 0 和 1
    # data = df.replace({"Label": {'BENIGN': 0, 'DDoS': 1}})
    # df.apply(pd.to_numeric, errors='ignore')
    print(df)
    features = [f for f in df.columns if f not in ['Timestamp', 'Label']]
    # 分割
    x_train = df[features]
    # object类型错误
    print(x_train.dtypes)
    print(x_train.columns)
    print(x_train["Dst Port"].unique())
    y_train = df['Label']
    y_pred = lgb_pre(x_train)
    # 新添加一列
    df['pred'] = y_pred
    # print(data)
    # 输出csv
    df.to_csv(outputname, index=False, sep=',')


def TestPre(path):
    df = pd.read_csv(path, skipinitialspace=True)
    # 处理缺失值及无穷
    df.replace((np.inf, -np.inf, np.nan), 0, inplace=True)
    # 替换标签值为 0 和 1
    df = df.replace({"Label": {'Benign': 0, 'DoS attacks-GoldenEye': 1, "DoS attacks-Slowloris": 1, 'DDOS attack-HOIC': 1, 'DDOS attack-LOIC-UDP': 1, 'DoS attacks-SlowHTTPTest': 1, 'DoS attacks-Hulk': 1}})
    features = [f for f in df.columns if f not in ['Timestamp', 'Label']]
    df = df.replace({"Label": {'BENIGN': 0, 'DDoS': 1}})
    features = [f for f in df.columns if f not in ['Timestamp', 'Label']]
    # 分割
    x_train = df[features]
    y_pred = lgb_pre_label(x_train)
    print(y_pred, list(df['Label']))
    accuracy = accuracy_score(list(df['Label']), list(y_pred))
    print("Accuracy: %.2f%%" % (accuracy * 100.0))


def TrainModel():
    """
    训练lgb 和 xgb模型
    :return:
    """
    # 载入数据首先
    path1 = "./MachineLearningCVE/Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv"
    path2 = "./MachineLearningCVE/Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv"
    path3 = "./MachineLearningCVE/Friday-16-02-2018_TrafficForML_CICFlowMeter.csv"
    df1 = pd.read_csv(path1, skipinitialspace=True)
    df2 = pd.read_csv(path2, skipinitialspace=True)
    df3 = pd.read_csv(path3, skipinitialspace=True)
    print(df3.columns)
    print(df3['Label'].unique())
    df = df1.append(df2, ignore_index=True)

    df = df.append(df3, ignore_index=True)
    # 处理缺失值及无穷
    df.replace((np.inf, -np.inf, np.nan), 0, inplace=True)
    # 替换标签值为 0 和 1
    data = df.replace({"Label": {'Benign': 0, 'DoS attacks-GoldenEye': 1, "DoS attacks-Slowloris": 1, 'DDOS attack-HOIC': 1, 'DDOS attack-LOIC-UDP': 1, 'DoS attacks-SlowHTTPTest': 1, 'DoS attacks-Hulk': 1}})
    features1 = [f for f in data.columns if f not in ['Timestamp', 'Label']]
    # 分割
    x_train = data[features1]
    y_train = data['Label']
    print("data", x_train, y_train)
    print(y_train.unique())

    lgb_trained_model, lgb_train = lgb_model(x_train, y_train)
    xgb_trained_model, xgb_train = xgb_model(x_train, y_train)


if __name__ == '__main__':
    # TrainModel()
    # OutputCsv(df, y_pred)
    OutputCsv2("./MachineLearningCVE/Friday-16-02-2018_TrafficForML_CICFlowMeter.csv", "Friday-16-02-2018_TrafficForML_CICFlowMeter_Pre.csv") # 不是这个
    OutputCsv2("./MachineLearningCVE/Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv", "Wednesday-21-02-2018_TrafficForML_CICFlowMeter_Pre.csv")
    OutputCsv2("./MachineLearningCVE/Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv", "Thursday-15-02-2018_TrafficForML_CICFlowMeter_Pre.csv")
    # OutputCsv2("./MachineLearningCVE/Friday-16-02-2018_TrafficForML_CICFlowMeter.csv", "Friday-16-02-2018_TrafficForML_CICFlowMeter_Pre.csv") # 奇怪的问题
    TestPre("./MachineLearningCVE/Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv")
    TestPre("./MachineLearningCVE/Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv")
    TestPre("./MachineLearningCVE/Friday-16-02-2018_TrafficForML_CICFlowMeter.csv")


