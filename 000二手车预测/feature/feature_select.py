#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/10/17 16:19
# @Author  : Wang Yuhang
# @File    : feature_select.py
# @Func    : 用于特征选择
import os
import numpy as np
import pandas as pd
import itertools
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


def build_model_lgb(x_train, y_train):
    model = lgb.LGBMRegressor(n_estimators=2000, num_leaves=90, max_depth=13)
    model.fit(x_train, y_train)
    return model


def run_model_lgb(train, price_log):
    x_train, x_val, y_train, y_val = train_test_split(
        train, price_log, test_size=0.3, random_state=2020)
    print('Train lgb ...')
    model_lgb = build_model_lgb(x_train, y_train)
    train_lgb = model_lgb.predict(x_train)
    train_mae_lgb = mean_absolute_error(np.exp(y_train), np.exp(train_lgb))
    print('MAE of train with lgb:', train_mae_lgb)
    val_lgb = model_lgb.predict(x_val)
    val_mae_lgb = mean_absolute_error(np.exp(y_val), np.exp(val_lgb))
    print('MAE of val with lgb:', val_mae_lgb)
    return model_lgb


def filter_corr(feature_group, corr, cutoff=0.7):
    cols = []
    for i, j in feature_group:                                              # i，j表示两个特征
        if corr.loc[i, j] > cutoff:                                         # corr.loc[i,j]表示i和j的相关系数
            # print(i, j, corr.loc[i, j])
            i_avg = corr[i][corr[i] != 1].mean()                            # i特征的相关度均值
            j_avg = corr[j][corr[j] != 1].mean()
            if i_avg >= j_avg:                                              # 留下相关度均值较大的特征
                cols.append(i)
            else:
                cols.append(j)
    return set(cols)


def select_cross_feature():
    # 筛选交叉特征中，重要度>5000的特征
    path = r'../data/data_for_select/'
    res_path = r'../data/feature_importance/'
    for file in os.listdir(path):
        print(file)
        data = pd.read_csv(path+file, sep=' ')
        feature_name = [f for f in data.columns.tolist() if f != 'price']
        price_log = data['price']
        train = data[feature_name]
        model_lgb = run_model_lgb(train, price_log)
        # 输出特征的重要程度
        feature_importance = [[name, value] for name, value in zip(train.columns, model_lgb.feature_importances_)]
        df = pd.DataFrame(feature_importance, columns=['feature', 'importance'])
        # a = df[df['importance'] > 5000]['feature'].values
        print(df)
        exit(0)
        df.to_csv(res_path+file, index=False)


select_cross_feature()

# 删除相关性高的变量
# data = pd.read_csv('../data/data_for_select/v_multiply_v.csv', sep=' ')
# corr = data.corr(method='spearman')                                         # 输出相关系数矩阵
# feature_group = list(itertools.combinations(corr.columns, 2))               # 输出corr.columns中元素组成的长度为2的子序列
# drop_cols = filter_corr(feature_group, corr, cutoff=0.95)


