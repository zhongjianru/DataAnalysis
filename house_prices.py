import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import skew
from scipy.stats import norm
from sklearn import preprocessing, linear_model, svm, gaussian_process
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder


# Kaggle：https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview
# 参考1：https://zhuanlan.zhihu.com/p/39429689
# 参考2：https://zhuanlan.zhihu.com/p/41761574

# 问题描述：基于竞赛方所提供的爱荷华州埃姆斯的住宅数据信息，预测每间房屋的销售价格。很明显这是一个回归问题！！

# 经过分析（列出所有列，区分数值型和分类型，画图），对于目标变量(Salesprices)有重要影响的四个变量，分别为
# OverallQual：总体评价（数值）
# YearBuilt：建造年份
# TotalBsmtSF：地下室总面积（数值）
# GrLivArea：生活面积（数值）


# 填充缺失值
# 一部分特征值数据的缺失是由于房屋确实不存在此种类型的特征，因此对于这一部分特征的缺失值，根据特征的数据类型分别进行插补：
# 类别特征的缺失值以一种新类别插补，数值特征以0值插补
# 剩余的那一部分缺失的特征值采用众数插补
def fill_missings(res):
    res['Alley'] = res['Alley'].fillna('missing')
    res['PoolQC'] = res['PoolQC'].fillna(res['PoolQC'].mode()[0])
    res['MasVnrType'] = res['MasVnrType'].fillna('None')
    res['BsmtQual'] = res['BsmtQual'].fillna(res['BsmtQual'].mode()[0])
    res['BsmtCond'] = res['BsmtCond'].fillna(res['BsmtCond'].mode()[0])
    res['FireplaceQu'] = res['FireplaceQu'].fillna(res['FireplaceQu'].mode()[0])
    res['GarageType'] = res['GarageType'].fillna('missing')
    res['GarageFinish'] = res['GarageFinish'].fillna(res['GarageFinish'].mode()[0])
    res['GarageQual'] = res['GarageQual'].fillna(res['GarageQual'].mode()[0])
    res['GarageCond'] = res['GarageCond'].fillna('missing')
    res['Fence'] = res['Fence'].fillna('missing')
    res['Street'] = res['Street'].fillna('missing')
    res['LotShape'] = res['LotShape'].fillna('missing')
    res['LandContour'] = res['LandContour'].fillna('missing')
    res['BsmtExposure'] = res['BsmtExposure'].fillna(res['BsmtExposure'].mode()[0])
    res['BsmtFinType1'] = res['BsmtFinType1'].fillna('missing')
    res['BsmtFinType2'] = res['BsmtFinType2'].fillna('missing')
    res['CentralAir'] = res['CentralAir'].fillna('missing')
    res['Electrical'] = res['Electrical'].fillna(res['Electrical'].mode()[0])
    res['MiscFeature'] = res['MiscFeature'].fillna('missing')
    res['MSZoning'] = res['MSZoning'].fillna(res['MSZoning'].mode()[0])
    res['Utilities'] = res['Utilities'].fillna('missing')
    res['Exterior1st'] = res['Exterior1st'].fillna(res['Exterior1st'].mode()[0])
    res['Exterior2nd'] = res['Exterior2nd'].fillna(res['Exterior2nd'].mode()[0])
    res['KitchenQual'] = res['KitchenQual'].fillna(res['KitchenQual'].mode()[0])
    res["Functional"] = res["Functional"].fillna("Typ")
    res['SaleType'] = res['SaleType'].fillna(res['SaleType'].mode()[0])
    # 数值型变量的空值先用0值替换
    flist = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
             'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea',
             'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
             'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
             'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']
    for fl in flist:
        res[fl] = res[fl].fillna(0)
    # 0值替换
    res['TotalBsmtSF'] = res['TotalBsmtSF'].apply(lambda x: np.exp(6) if x <= 0.0 else x)
    res['2ndFlrSF'] = res['2ndFlrSF'].apply(lambda x: np.exp(6.5) if x <= 0.0 else x)
    res['GarageArea'] = res['GarageArea'].apply(lambda x: np.exp(6) if x <= 0.0 else x)
    res['GarageCars'] = res['GarageCars'].apply(lambda x: 0 if x <= 0.0 else x)
    res['LotFrontage'] = res['LotFrontage'].apply(lambda x: np.exp(4.2) if x <= 0.0 else x)
    res['MasVnrArea'] = res['MasVnrArea'].apply(lambda x: np.exp(4) if x <= 0.0 else x)
    res['BsmtFinSF1'] = res['BsmtFinSF1'].apply(lambda x: np.exp(6.5) if x <= 0.0 else x)
    return res


# 对于顺序变量，标签编码（LabelEncoder）的方式无法正确识别这种顺序关系，因此这里通过自定义函数实现编码转换
def trans_label(my_data):
    my_data['TotalSF'] = my_data['TotalBsmtSF'] + my_data['1stFlrSF'] + my_data['2ndFlrSF']

    def QualToInt(x):
        if (x == 'Ex'):
            r = 0
        elif (x == 'Gd'):
            r = 1
        elif (x == 'TA'):
            r = 2
        elif (x == 'Fa'):
            r = 3
        elif (x == 'missing'):
            r = 4
        else:
            r = 5
        return r

    my_data['ExterQual'] = my_data['ExterQual'].apply(QualToInt)
    my_data['ExterCond'] = my_data['ExterCond'].apply(QualToInt)
    my_data['KitchenQual'] = my_data['KitchenQual'].apply(QualToInt)
    my_data['HeatingQC'] = my_data['HeatingQC'].apply(QualToInt)
    my_data['BsmtQual'] = my_data['BsmtQual'].apply(QualToInt)
    my_data['BsmtCond'] = my_data['BsmtCond'].apply(QualToInt)
    my_data['FireplaceQu'] = my_data['FireplaceQu'].apply(QualToInt)
    my_data['GarageQual'] = my_data['GarageQual'].apply(QualToInt)
    my_data['PoolQC'] = my_data['PoolQC'].apply(QualToInt)

    return my_data


def addlogs(res, ls):
    m = res.shape[1]
    for l in ls:
        res = res.assign(newcol=pd.Series(np.log(1.01 + res[l])).values)
        res.columns.values[m] = l + '_log'
        m += 1
    return res


def getdummies(res, ls):
    def encode(encode_df):
        encode_df = np.array(encode_df)
        enc = OneHotEncoder()
        le = LabelEncoder()
        le.fit(encode_df)
        res1 = le.transform(encode_df).reshape(-1, 1)
        enc.fit(res1)
        return pd.DataFrame(enc.transform(res1).toarray()), le, enc

    decoder = []
    outres = pd.DataFrame({'A': []})

    for l in ls:
        cat, le, enc = encode(res[l])
        cat.columns = [l + str(x) for x in cat.columns]
        outres.reset_index(drop=True, inplace=True)
        outres = pd.concat([outres, cat], axis=1)
        decoder.append([le, enc])
    return (outres, decoder)


def prices_f1():
    # 进一步展示房屋售价
    # figure = plt.figure()
    sns.pairplot(x_vars=['OverallQual', 'GrLivArea', 'YearBuilt', 'TotalBsmtSF'], y_vars=['SalePrice'], data=train_data,
                 dropna=True)  # 散点图
    # sns.distplot(train_data['SalePrice'])  # 柱状图
    plt.show()

    # 通过散点图的方式可以观察到一些可疑的异常值，对不合理的离群值考虑将其删除
    train_data.drop(train_data[(train_data['OverallQual'] < 5) & (train_data['SalePrice'] > 200000)].index,
                    inplace=True)
    train_data.drop(train_data[(train_data['GrLivArea'] > 4000) & (train_data['SalePrice'] < 200000)].index,
                    inplace=True)
    train_data.drop(train_data[(train_data['YearBuilt'] < 1900) & (train_data['SalePrice'] > 400000)].index,
                    inplace=True)
    train_data.drop(train_data[(train_data['TotalBsmtSF'] > 6000) & (train_data['SalePrice'] < 200000)].index,
                    inplace=True)
    train_data.reset_index(drop=True, inplace=True)

    # 将训练数据集和测试数据集合并为一个数据集，在最后对模型进行训练时再将合并的数据集按照索引重新分割为训练集和测试集
    my_data = pd.concat([train_data, test_data], axis=0)
    my_data.reset_index(drop=True, inplace=True)
    train_index = train_data.index
    test_index = list(set(my_data.index).difference(set(train_data.index)))

    # 处理缺失数据
    # 先查看数据的缺失情况
    all_data = pd.concat([train_data, test_data])
    count = all_data.isnull().sum().sort_values(ascending=False)
    ratio = count / len(all_data)
    nulldata = pd.concat([count, ratio], axis=1, keys=['count', 'ratio'])
    print(nulldata)
    # 填充缺失数据
    my_data = fill_missings(my_data)

    # 观察变量之间的相关性
    corrmat = train_data.corr()
    plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=0.9, square=True)
    plt.show()

    # 一些特征其被表示成数值特征缺乏意义，例如年份还有类别，这里将其转换为字符串，即类别型变量
    my_data['MSSubClass'] = my_data['MSSubClass'].apply(str)
    my_data['YrSold'] = my_data['YrSold'].astype(str)
    my_data['MoSold'] = my_data['MoSold'].astype(str)
    all_data['OverallCond'] = all_data['OverallCond'].astype(str)

    # 转换顺序特征编码
    my_data = train_data(my_data)

    # 增添特征
    # 由于区域相关特征对于确定房价非常重要，我们还增加了一个特征，即每个房屋的地下室，一楼和二楼的总面积
    my_data['TotalSF'] = my_data['TotalBsmtSF'] + my_data['1stFlrSF'] + my_data['2ndFlrSF']
    # 房屋内某些区域空间的有无通常也是影响房屋价格的重要因素，所以增添几个特征用于描述房屋内是否存在这些区域空间
    my_data['HasWoodDeck'] = (my_data['WoodDeckSF'] == 0) * 1
    my_data['HasOpenPorch'] = (my_data['OpenPorchSF'] == 0) * 1
    my_data['HasEnclosedPorch'] = (my_data['EnclosedPorch'] == 0) * 1
    my_data['Has3SsnPorch'] = (my_data['3SsnPorch'] == 0) * 1
    my_data['HasScreenPorch'] = (my_data['ScreenPorch'] == 0) * 1
    # 房屋改造时间（YearsSinceRemodel）与房屋出售时间（YrSold）间隔时间的长短通常也会影响房价
    my_data['YearsSinceRemodel'] = my_data['YrSold'].astype(int) - my_data['YearRemodAdd'].astype(int)
    # 房屋的整体质量也是影响房价的重要因素
    my_data['Total_Home_Quality'] = my_data['OverallQual'] + my_data['OverallCond']

    # 数据转换：比较常用的有对数转换，box-cox转换等变换方式
    # 其中对数转换的方式是最为常用的，取对数之后数据的性质和相关关系不会发生改变，但压缩了变量的尺度，大大方便了计算
    # 绘制每个定量变量与目标变量的分布图
    quantitative = [f for f in train_data.columns if train_data.dtypes[f] != 'object' and train_data.dtypes[f] != 'str']
    quantitative.remove('SalePrice')
    f = pd.melt(train_data, value_vars=quantitative)
    g = sns.FacetGrid(f, col="variable", col_wrap=5, sharex=False, sharey=False)
    g = g.map(sns.distplot, "value")
    plt.show()

    # 计算各定量变量的偏度（skewness）
    skewed_feats = my_data[quantitative].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    skewness = pd.DataFrame({'Skew': skewed_feats})
    print(skewness.head(20))

    # 每个定量变量都有不同程度的偏移，对于偏度大于0.15的定量变量，可以对其进行log操作以提升质量
    loglist = skewness[abs(skewness) > 0.15].index.tolist()
    my_data = addlogs(my_data, loglist)

    # 找出所有需要编码的定性变量，这里记得要除去已经做了转换的顺序变量
    qualitative = [f for f in train_data.columns if train_data.dtypes[f] == 'object' or train_data.dtypes[f] == 'str']
    oridnals = ['BsmtFinType1', 'MasVnrType', 'Foundation', 'HouseStyle', 'Functional', 'BsmtExposure', 'GarageFinish',
                'PavedDrive', 'Street',
                'ExterQual', 'PavedDrive', 'ExterQua', 'ExterCond', 'KitchenQual', 'HeatingQC', 'BsmtQual',
                'FireplaceQu', 'GarageQual', 'PoolQC']
    qualitative = list(set(qualitative).difference(set(oridnals)))

    # 对定性编码进行独热编码，并与之前的顺序变量以及已经做了log转换的定量变量合并为一个数据集
    catpredlist = qualitative
    res = getdummies(my_data[catpredlist], catpredlist)
    df = res[0]
    decoder = res[1]
    floatAndordinal = list(set(my_data.columns.values).difference(set(qualitative)))
    my_data.columns.values
    print(df.shape)
    df = pd.concat([df, my_data[floatAndordinal]], axis=1)
    df.drop(['SalePrice'], axis=1, inplace=True)

    # 特征降维：简单的PCA降维处理
    pca = PCA(n_components=295)
    df = pd.DataFrame(pca.fit_transform(df))

    # 训练模型
    # 先把合并的数据集重新分割为训练集和测试集
    df_train = df.iloc[train_index]
    df_score = df.iloc[test_index]
    my_traindata = my_data.iloc[train_index]
    X = np.array(df_train)
    X = np.delete(X, 0, 1)
    y = np.log(1 + np.array(my_traindata['SalePrice']))
    print(X.shape)

    # 建模：house_model.py


def prices_f2():
    # 整体关系矩阵，深色是负相关，浅色是正相关，查看某两列的相关性
    # 只有数值型变量可以绘制关系矩阵，把离散型变量的数据处理成数值型
    # 可以看出上面提到的四个变量跟Saleprices的相关性最大
    cols = ['CentralAir', 'Neighborhood']  # 离散型变量
    for x in cols:
        label = preprocessing.LabelEncoder()
        train_data[x] = label.fit_transform(train_data[x])
    corrmat = train_data.corr()
    f, ax = plt.subplots(figsize=(20, 9))
    sns.heatmap(corrmat, vmax=1, square=True)
    # plt.show()

    # 房价关系矩阵
    k = 10
    cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index  # 取前10个相关性最大的特征cols.values.tolist()
    cm = np.corrcoef(train_data[cols].values.T)
    sns.set(font_scale=1.25)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10},
                     yticklabels=cols.values, xticklabels=cols.values)
    # plt.show()

    # 选取上面相关性较大，并且比较具有代表性的变量（比如'GarageCars'和'GarageArea'留一个就够了），绘制任意两个变量之间的图表
    # 房屋售价、总体评价、生活面积、车库、地下室总面积、浴室数量、总房间数、建造年份
    cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd',
            'YearBuilt']
    sns.pairplot(train_data[cols], height=2.5)
    # plt.show()

    # 开始模拟数据
    x = train_data[cols].drop(columns='SalePrice').values
    y = train_data['SalePrice'].values
    x_scaled = preprocessing.StandardScaler().fit_transform(x)
    y_scaled = preprocessing.StandardScaler().fit_transform(y.reshape(-1, 1))
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.33, random_state=42)
    clfs = {
        'svm': svm.SVR(),
        'RandomForestRegressor': RandomForestRegressor(n_estimators=400),
        'BayesianRidge': linear_model.BayesianRidge()
    }
    for clf in clfs:
        try:
            clfs[clf].fit(x_train, y_train)
            y_pred = clfs[clf].predict(x_test)
            # print(clf, 'cost:', str(np.sum(y_pred-y_test)/len(y_pred)))
        except Exception as e:
            print(clf, 'Error:', e)

    # 由上面结果选择随机森林回归算法
    # 为了更直观地观察训练结果，展示一下未归一化数据的预测效果

    # 查看数据缺失情况
    print(train_data[cols].isnull().sum())
    # print(train_data['SalePrice'].describe())


if __name__ == '__main__':
    pd.set_option('display.width', 1000)
    # pd.set_option('display.max_cols', 1000)
    train_data = pd.read_csv('house_prices/train.csv')
    test_data = pd.read_csv('house_prices/test.csv')
    prices_f1()
