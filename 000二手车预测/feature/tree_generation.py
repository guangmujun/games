#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/10/15 10:22
# @Author  : Wang Yuhang
# @File    : tree_generation.py
# @Func    :

# 导入包
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# DataFrame显示设置
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.max_rows', None)     # 显示所有行
pd.set_option('max_colwidth', 100)          # 设置value的显示长度为100，默认为50

# 导入数据
Train_data = pd.read_csv('../data/train.csv', sep=' ')
Test_data = pd.read_csv('../data/test.csv', sep=' ')
print(Train_data.shape, Test_data.shape)

"""
---------------------------以下为树模型的数据处理----------------------------
"""
"""
一、预测值处理，处理目标值长尾分布问题
"""
Train_data['price'] = np.log1p(Train_data['price'])             # 对price做log变换

# 合并，方便后续的操作
df = pd.concat([Train_data, Test_data], ignore_index=True)     # 合并train和test，一起处理

"""
二、数据简单预处理，分三步进行
"""
# 1.处理无用值和基本无变化的值
df['name_count'] = df.groupby(['name'])['SaleID'].transform(
    'count')  # 统计不同name的SaleID的数量
del df['name']

df.drop(df[df['seller'] == 1].index, inplace=True)
del df['offerType']
del df['seller']

# 2.处理缺失值
df['fuelType'] = df['fuelType'].fillna(0)
df['gearbox'] = df['gearbox'].fillna(0)
df['bodyType'] = df['bodyType'].fillna(0)
df['model'] = df['model'].fillna(0)

# 3. 处理异常值
df['power'] = df['power'].map(lambda x: 600 if x > 600 else x)
df['notRepairedDamage'] = df['notRepairedDamage'].astype(
    'str').apply(lambda x: x if x != '-' else None).astype('float32')

"""
三、特征工程
"""
# 1. 时间特征处理


def data_process(x):
    year = int(str(x)[:4])
    month = int(str(x)[4:6])
    day = int(str(x)[6:8])

    if month < 1:
        month = 1

    date = datetime(year, month, day)
    return date


df['regDate'] = df['regDate'].apply(data_process)
df['creatDate'] = df['creatDate'].apply(data_process)
df['regDate_year'] = df['regDate'].dt.year
df['regDate_month'] = df['regDate'].dt.month
df['regDate_day'] = df['regDate'].dt.day
df['creatDate_year'] = df['creatDate'].dt.year
df['creatDate_month'] = df['creatDate'].dt.month
df['creatDate_day'] = df['creatDate'].dt.day
df['car_age_day'] = (df['creatDate'] - df['regDate']).dt.days
df['car_year_year'] = round(df['car_age_day'] / 365, 1)

# 2. 地区特征处理
df['regionCode_count'] = df.groupby(
    ['regionCode'])['SaleID'].transform('count')
df['city'] = df['regionCode'].apply(lambda x: str(x)[:2])

# 3. 分类特征处理
# 指定区间，power分为30个桶
bin = [i * 10 for i in range(31)]
df['power_bin'] = pd.cut(df['power'], bin, labels=False)

bin = [i * 10 for i in range(24)]
df['model_bin'] = pd.cut(df['model'], bin, labels=False)

# 基础特征输出
df.to_csv('../data/data_for_select/base_feature.csv', index=False, sep=' ')
print('--base_feature已输出--')

# 特征交叉


def output_cross_feature(brand_fe, df, name='result'):
    # 输出构造的交叉特征，用于特征选择
    features = brand_fe.columns.tolist() + ['price']
    df[features][:50000].to_csv('../data/data_for_select/'+name+'.csv', index=False, sep=' ')
    print('--{name}已输出--'.format(name=name))


Train_gb = Train_data.groupby('regionCode')
all_info = {}
for kind, kind_data in Train_gb:                                # kind表示regionCode中的类型，kind_data表示对应类型的数据
    info = {}
    kind_data = kind_data[kind_data['price'] > 0]
    info['regionCode_amount'] = len(kind_data)
    info['regionCode_price_max'] = kind_data.price.max()
    info['regionCode_price_min'] = kind_data.price.min()
    info['regionCode_price_median'] = kind_data.price.median()
    info['regionCode_price_sum'] = kind_data.price.sum()
    info['regionCode_price_std'] = kind_data.price.std()
    info['regionCode_price_mean'] = kind_data.price.mean()
    info['regionCode_price_skew'] = kind_data.price.skew()
    info['regionCode_price_kurt'] = kind_data.price.kurt()
    # Return the mean absolute deviation
    info['regionCode_mad'] = kind_data.price.mad()
    all_info[kind] = info
brand_fe = pd.DataFrame(all_info).T.reset_index().rename(
    columns={'index': 'regionCode'})  # 每一列表示一个特征
# left保留左表的所有数据，on指定用于对齐的列名
df = df.merge(brand_fe, how='left', on='regionCode')
# 输出构造的交叉特征，用于特征选择
output_cross_feature(brand_fe, df, name='regionCode_price')

Train_gb = Train_data.groupby("brand")
all_info = {}
for kind, kind_data in Train_gb:
    info = {}
    kind_data = kind_data[kind_data['price'] > 0]
    info['brand_amount'] = len(kind_data)
    info['brand_price_max'] = kind_data.price.max()
    info['brand_price_median'] = kind_data.price.median()
    info['brand_price_min'] = kind_data.price.min()
    info['brand_price_sum'] = kind_data.price.sum()
    info['brand_price_std'] = kind_data.price.std()
    info['brand_price_mean'] = kind_data.price.mean()
    info['brand_price_skew'] = kind_data.price.skew()
    info['brand_price_kurt'] = kind_data.price.kurt()
    info['brand_price_mad'] = kind_data.price.mad()
    all_info[kind] = info
brand_fe = pd.DataFrame(all_info).T.reset_index().rename(
    columns={"index": "brand"})
df = df.merge(brand_fe, how='left', on='brand')
# 输出构造的交叉特征，用于特征选择
output_cross_feature(brand_fe, df, name='brand_price')

Train_gb = Train_data.groupby("model")
all_info = {}
for kind, kind_data in Train_gb:
    info = {}
    kind_data = kind_data[kind_data['price'] > 0]
    info['model_amount'] = len(kind_data)
    info['model_price_max'] = kind_data.price.max()
    info['model_price_median'] = kind_data.price.median()
    info['model_price_min'] = kind_data.price.min()
    info['model_price_sum'] = kind_data.price.sum()
    info['model_price_std'] = kind_data.price.std()
    info['model_price_mean'] = kind_data.price.mean()
    info['model_price_skew'] = kind_data.price.skew()
    info['model_price_kurt'] = kind_data.price.kurt()
    info['model_price_mad'] = kind_data.price.mad()
    all_info[kind] = info
brand_fe = pd.DataFrame(all_info).T.reset_index().rename(
    columns={"index": "model"})
df = df.merge(brand_fe, how='left', on='model')
# 输出构造的交叉特征，用于特征选择
output_cross_feature(brand_fe, df, name='model_price')

Train_gb = Train_data.groupby("kilometer")
all_info = {}
for kind, kind_data in Train_gb:
    info = {}
    kind_data = kind_data[kind_data['price'] > 0]
    info['kilometer_amount'] = len(kind_data)
    info['kilometer_price_max'] = kind_data.price.max()
    info['kilometer_price_median'] = kind_data.price.median()
    info['kilometer_price_min'] = kind_data.price.min()
    info['kilometer_price_sum'] = kind_data.price.sum()
    info['kilometer_price_std'] = kind_data.price.std()
    info['kilometer_price_mean'] = kind_data.price.mean()
    info['kilometer_price_skew'] = kind_data.price.skew()
    info['kilometer_price_kurt'] = kind_data.price.kurt()
    info['kilometer_price_mad'] = kind_data.price.mad()
    all_info[kind] = info
brand_fe = pd.DataFrame(all_info).T.reset_index().rename(
    columns={"index": "kilometer"})
df = df.merge(brand_fe, how='left', on='kilometer')
# 输出构造的交叉特征，用于特征选择
output_cross_feature(brand_fe, df, name='kilometer_price')


Train_gb = Train_data.groupby("bodyType")
all_info = {}
for kind, kind_data in Train_gb:
    info = {}
    kind_data = kind_data[kind_data['price'] > 0]
    info['bodyType_amount'] = len(kind_data)
    info['bodyType_price_max'] = kind_data.price.max()
    info['bodyType_price_median'] = kind_data.price.median()
    info['bodyType_price_min'] = kind_data.price.min()
    info['bodyType_price_sum'] = kind_data.price.sum()
    info['bodyType_price_std'] = kind_data.price.std()
    info['bodyType_price_mean'] = kind_data.price.mean()
    info['bodyType_price_skew'] = kind_data.price.skew()
    info['bodyType_price_kurt'] = kind_data.price.kurt()
    info['bodyType_price_mad'] = kind_data.price.mad()
    all_info[kind] = info
brand_fe = pd.DataFrame(all_info).T.reset_index().rename(
    columns={"index": "bodyType"})
df = df.merge(brand_fe, how='left', on='bodyType')
# 输出构造的交叉特征，用于特征选择
output_cross_feature(brand_fe, df, name='bodyType_price')

Train_gb = Train_data.groupby("fuelType")
all_info = {}
for kind, kind_data in Train_gb:
    info = {}
    kind_data = kind_data[kind_data['price'] > 0]
    info['fuelType_amount'] = len(kind_data)
    info['fuelType_price_max'] = kind_data.price.max()
    info['fuelType_price_median'] = kind_data.price.median()
    info['fuelType_price_min'] = kind_data.price.min()
    info['fuelType_price_sum'] = kind_data.price.sum()
    info['fuelType_price_std'] = kind_data.price.std()
    info['fuelType_price_mean'] = kind_data.price.mean()
    info['fuelType_price_skew'] = kind_data.price.skew()
    info['fuelType_price_kurt'] = kind_data.price.kurt()
    info['fuelType_price_mad'] = kind_data.price.mad()
    all_info[kind] = info
brand_fe = pd.DataFrame(all_info).T.reset_index().rename(
    columns={"index": "fuelType"})
df = df.merge(brand_fe, how='left', on='fuelType')
# 输出构造的交叉特征，用于特征选择
output_cross_feature(brand_fe, df, name='fuelType_price')

kk = "regionCode"
Train_gb = df.groupby(kk)
all_info = {}
for kind, kind_data in Train_gb:
    info = {}
    kind_data = kind_data[kind_data['car_age_day'] > 0]             # 地区与车龄
    info[kk + '_days_max'] = kind_data.car_age_day.max()
    info[kk + '_days_min'] = kind_data.car_age_day.min()
    info[kk + '_days_std'] = kind_data.car_age_day.std()
    info[kk + '_days_mean'] = kind_data.car_age_day.mean()
    info[kk + '_days_median'] = kind_data.car_age_day.median()
    info[kk + '_days_sum'] = kind_data.car_age_day.sum()
    all_info[kind] = info
brand_fe = pd.DataFrame(all_info).T.reset_index().rename(columns={"index": kk})
df = df.merge(brand_fe, how='left', on=kk)
# 输出构造的交叉特征，用于特征选择
output_cross_feature(brand_fe, df, name='regionCode_car_age_day')

Train_gb = df.groupby(kk)
all_info = {}
for kind, kind_data in Train_gb:
    info = {}
    kind_data = kind_data[kind_data['power'] > 0]                  # 地区与马力
    info[kk + '_power_max'] = kind_data.power.max()
    info[kk + '_power_min'] = kind_data.power.min()
    info[kk + '_power_std'] = kind_data.power.std()
    info[kk + '_power_mean'] = kind_data.power.mean()
    info[kk + '_power_median'] = kind_data.power.median()
    info[kk + '_power_sum'] = kind_data.power.sum()
    all_info[kind] = info
brand_fe = pd.DataFrame(all_info).T.reset_index().rename(columns={"index": kk})
df = df.merge(brand_fe, how='left', on=kk)
# 输出构造的交叉特征，用于特征选择
output_cross_feature(brand_fe, df, name='regionCode_power')

# 3. 连续数值特征处理
dd = 'v_3'
Train_gb = df.groupby(kk)
all_info = {}
for kind, kind_data in Train_gb:
    info = {}
    kind_data = kind_data[kind_data[dd] > -10000000]
    info[kk + '_' + dd + '_max'] = kind_data.v_3.max()
    info[kk + '_' + dd + '_min'] = kind_data.v_3.min()
    info[kk + '_' + dd + '_std'] = kind_data.v_3.std()
    info[kk + '_' + dd + '_mean'] = kind_data.v_3.mean()
    info[kk + '_' + dd + '_median'] = kind_data.v_3.median()
    info[kk + '_' + dd + '_sum'] = kind_data.v_3.sum()
    all_info[kind] = info
brand_fe = pd.DataFrame(all_info).T.reset_index().rename(columns={"index": kk})
df = df.merge(brand_fe, how='left', on=kk)
# 输出构造的交叉特征，用于特征选择
output_cross_feature(brand_fe, df, name='regionCode_v_3')

dd = 'v_0'
Train_gb = df.groupby(kk)
all_info = {}
for kind, kind_data in Train_gb:
    info = {}
    kind_data = kind_data[kind_data[dd] > -10000000]
    info[kk + '_' + dd + '_max'] = kind_data.v_0.max()
    info[kk + '_' + dd + '_min'] = kind_data.v_0.min()
    info[kk + '_' + dd + '_std'] = kind_data.v_0.std()
    info[kk + '_' + dd + '_mean'] = kind_data.v_0.mean()
    info[kk + '_' + dd + '_median'] = kind_data.v_0.median()
    info[kk + '_' + dd + '_sum'] = kind_data.v_0.sum()
    all_info[kind] = info
brand_fe = pd.DataFrame(all_info).T.reset_index().rename(columns={"index": kk})
df = df.merge(brand_fe, how='left', on=kk)
# 输出构造的交叉特征，用于特征选择
output_cross_feature(brand_fe, df, name='regionCode_v_0')


"""
四、补充的特征工程
"""
# 主要是对匿名特征和几个重要度较高的分类特征进行特征交叉
# 第一批特征工程
feature_for_select = []
for i in range(15):
    for j in range(15):
        df['new'+str(i)+'*'+str(j)] = df['v_'+str(i)]*df['v_'+str(j)]
        feature_for_select.append('new'+str(i)+'*'+str(j))
# 输出构造的交叉特征，用于特征选择
feature_for_select += ['price']
df[feature_for_select][:50000].to_csv('../data/data_for_select/v_multiply_v.csv', index=False, sep=' ')
print('--v_multiply_v已输出--')

# 第二批特征工程
feature_for_select = []
for i in range(15):
    for j in range(15):
        df['new'+str(i)+'+'+str(j)] = df['v_'+str(i)]+df['v_'+str(j)]
        feature_for_select.append('new' + str(i) + '+' + str(j))
# 输出构造的交叉特征，用于特征选择
feature_for_select += ['price']
df[feature_for_select][:50000].to_csv('../data/data_for_select/v_plus_v.csv', index=False, sep=' ')
print('--v_plus_v已输出--')

exit(0)

"""
五、筛选特征
"""
numerical_cols = df.select_dtypes(exclude='object').columns
list_tree = ['model_power_sum', 'price', 'SaleID',
             'model_power_std', 'model_power_median', 'model_power_max',
             'brand_price_max', 'brand_price_median',
             'brand_price_sum', 'brand_price_std',
             'model_days_sum',
             'model_days_std', 'model_days_median', 'model_days_max', 'model_bin', 'model_amount',
             'model_price_max', 'model_price_median',
             'model_price_min', 'model_price_sum', 'model_price_std',
             'model_price_mean', 'bodyType', 'model', 'brand', 'fuelType', 'gearbox', 'power', 'kilometer',
             'notRepairedDamage', 'v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10',
             'v_11', 'v_12', 'v_13', 'v_14', 'name_count', 'regDate_year', 'car_age_day', 'car_age_year',
             'power_bin', 'fuelType', 'gearbox', 'kilometer', 'notRepairedDamage', 'name_count', 'car_age_day',
             'new3*3', 'new12*14', 'new2*14', 'new14*14']

for i in range(15):
    for j in range(15):
        list_tree.append('new' + str(i) + '+' + str(j))

feature_cols = [col for col in numerical_cols if col in list_tree]

feature_cols = [col for col in feature_cols if col not in
                ['new14+6', 'new13+6', 'new0+12', 'new9+11', 'v_3', 'new11+10', 'new10+14', 'new12+4', 'new3+4',
                 'new11+11', 'new13+3', 'new8+1', 'new1+7', 'new11+14', 'new8+13', 'v_8', 'v_0', 'new3+5',
                 'new2+9', 'new9+2', 'new0+11', 'new13+7', 'new8+11', 'new5+12', 'new10+10', 'new13+8',
                 'new11+13', 'new7+9', 'v_1', 'new7+4', 'new13+4', 'v_7', 'new5+6', 'new7+3', 'new9+10', 'new11+12',
                 'new0+5', 'new4+13', 'new8+0', 'new0+7', 'new12+8', 'new10+8', 'new13+14', 'new5+7', 'new2+7', 'v_4',
                 'v_10', 'new4+8', 'new8+14', 'new5+9', 'new9+13', 'new2+12', 'new5+8', 'new3+12', 'new0+10', 'new9+0',
                 'new1+11', 'new8+4', 'new11+8', 'new1+1', 'new10+5', 'new8+2', 'new6+1', 'new2+1', 'new1+12', 'new2+5',
                 'new0+14', 'new4+7', 'new14+9', 'new0+2', 'new4+1', 'new7+11', 'new13+10', 'new6+3', 'new1+10', 'v_9',
                 'new3+6', 'new12+1', 'new9+3', 'new4+5', 'new12+9', 'new3+8', 'new0+8', 'new1+8', 'new1+6', 'new10+9',
                 'new5+4', 'new13+1', 'new3+7', 'new6+4', 'new6+7', 'new13+0', 'new1+14', 'new3+11', 'new6+8', 'new0+9',
                 'new2+14', 'new6+2', 'new12+12', 'new7+12', 'new12+6', 'new12+14', 'new4+10', 'new2+4', 'new6+0',
                 'new3+9', 'new2+8', 'new6+11', 'new3+10', 'new7+0', 'v_11', 'new1+3', 'new8+3', 'new12+13', 'new1+9',
                 'new10+13', 'new5+10', 'new2+2', 'new6+9', 'new7+10', 'new0+0', 'new11+7', 'new2+13', 'new11+1',
                 'new5+11', 'new4+6', 'new12+2', 'new4+4', 'new6+14', 'new0+1', 'new4+14', 'v_5', 'new4+11', 'v_6',
                 'new0+4', 'new1+5', 'new3+14', 'new2+10', 'new9+4', 'new2+6', 'new14+14', 'new11+6', 'new9+1',
                 'new3+13', 'new13+13', 'new10+6', 'new2+3', 'new2+11', 'new1+4', 'v_2', 'new5+13', 'new4+2', 'new0+6',
                 'new7+13', 'new8+9', 'new9+12', 'new0+13', 'new10+12', 'new5+14', 'new6+10', 'new10+7', 'v_13',
                 'new5+2', 'new6+13', 'new9+14', 'new13+9', 'new14+7', 'new8+12', 'new3+3', 'new6+12', 'v_12',
                 'new14+4', 'new11+9', 'new12+7', 'new4+9', 'new4+12', 'new1+13', 'new0+3', 'new8+10', 'new13+11',
                 'new7+8', 'new7+14', 'v_14', 'new10+11', 'new14+8', 'new1+2']]

df = df[feature_cols]

"""
六、导出数据
"""
print(df.shape)
train_num = df.shape[0] - 50000
df[0:int(train_num)].to_csv('../data/train_tree.csv', index=False, sep=' ')
df[train_num:train_num +
    50000].to_csv('../data/test_tree.csv', index=False, sep=' ')
print('--树模型数据已经准备完毕--')
