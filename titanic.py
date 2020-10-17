# Kaggle: https://www.kaggle.com/c/titanic/rules
# Course: https://www.jianshu.com/p/e79a8c41cb1a、https://mp.weixin.qq.com/s/qa7NlMsDvLu6SZY0ZL2LCA

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, os
from six.moves import urllib
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn import metrics


def set_option():
    warnings.filterwarnings('ignore')  # 利用过滤器来忽略告警
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_columns', 1000)
    pd.set_option('display.max_colwidth', 100)
    pd.set_option('display.max_rows', 2000)


def load_data():
    url = 'https://raw.githubusercontent.com/hitcszq/kaggle_titanic/master/'
    path = 'titanic/'
    train_name = 'train.csv'
    test_name = 'test.csv'
    titanic_name = 'titanic.csv'

    if not os.path.isdir(path):
        os.mkdir(path)
        urllib.request.urlretrieve(os.path.join(url, train_name), os.path.join(path, train_name))
        urllib.request.urlretrieve(os.path.join(url, test_name), os.path.join(path, test_name))
        urllib.request.urlretrieve(os.path.join(url, titanic_name), os.path.join(path, titanic_name))

    train = pd.read_csv(os.path.join(path, train_name))
    test = pd.read_csv(os.path.join(path, test_name))
    PassengerId = test['PassengerId']
    all_data = pd.concat([train, test], ignore_index=True)
    return train, test, PassengerId, all_data


# 1、总体预览
def s1_preview(data):
    print(data.info())  # 查看各列数据情况
    print(data.describe())  # 各列统计值，最值、平均值等


# 2、数据初步分析（针对 train） & 特征值补充（针对 all_data）
def s2_preliminary(train, all_data):
    # 1、初步了解数据之间的相关性：是否生还与乘客性别、年龄、仓位相关性最大
    print(train['Survived'].value_counts())  # 幸存人数
    print(train.groupby(['Sex', 'Pclass'])['Survived'].value_counts())
    print(train.groupby(['Sex', 'Pclass'])['Survived'].aggregate('mean'))  # 与性别、仓位的关系
    print(train.groupby(['Sex', 'Pclass'])['Survived'].aggregate('mean').unstack())
    print(train.pivot_table('Survived', index='Sex', columns='Pclass', aggfunc='mean'))  # unstack 与数据透视图效果等价

    age = pd.cut(train['Age'], [0, 18, 50, 80])  # 年龄分段
    print(train.pivot_table('Survived', index=['Sex', age], columns='Pclass', aggfunc='mean'))

    # 绘图：（barplot 的默认估计值为平均值，无需指定）
    sns.barplot(x='Sex', y='Survived', data=train)  # 性别
    sns.barplot(x='Pclass', y='Survived', data=train)  # 仓位
    sns.barplot(x='SibSp', y='Survived', data=train)  # 配偶及兄弟姐妹数
    sns.barplot(x='Parch', y='Survived', data=train)  # 父母及子女数
    sns.countplot(x='Embarked', hue='Survived', data=train)  # 登船港口（相当于：group by Embarked 分别统计 Survived 生还 / 不生还人数）
    # plt.show()  # 一类图像只会出现最后一个

    # 不同年龄生还情况密度图：在 0-12 岁、12-30 岁之间，生还率有显著差异，其他年龄段差异不明显
    facet = sns.FacetGrid(train, hue='Survived', aspect=2)
    facet.map(sns.kdeplot, 'Age', shade=True)
    facet.set(xlim=(0, train['Age'].max()))
    facet.add_legend()
    plt.xlabel('Age')
    plt.ylabel('density')
    # plt.show()

    # 2、补充特征值：不同称呼的乘客生还情况：新增 Title 特征，从姓名中提取乘客的称呼，并归纳为 6 类
    all_data['Title'] = all_data['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
    title_dict = {}
    title_dict.update(dict.fromkeys(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'))
    title_dict.update(dict.fromkeys(['Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty'))
    title_dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 'Mrs'))
    title_dict.update(dict.fromkeys(['Mlle', 'Miss'], 'Miss'))
    title_dict.update(dict.fromkeys(['Mr'], 'Mr'))
    title_dict.update(dict.fromkeys(['Master', 'Jonkheer'], 'Master'))
    all_data['Title'] = all_data['Title'].map(title_dict)  # 对称呼进行转换
    sns.barplot(x='Title', y='Survived', data=all_data)
    # plt.show()

    # 补充特征值：不同姓氏的乘客生还情况：从姓名中提取乘客的姓氏，从人数大于 1 的组中分别提取出每组的妇女儿童和成年男性人数
    all_data['Surname'] = all_data['Name'].apply(lambda x: x.split(',')[0].strip())
    surname_count = all_data['Surname'].value_counts().to_dict()
    surname_mean = all_data.groupby('Surname')['Survived'].mean().to_dict()
    # 家庭成员人数
    all_data['FamilyGroup'] = all_data['Surname'].apply(lambda x: surname_count[x])  # 这里的 apply 与上面用到的 map 等价
    # 家庭成员幸存率
    all_data['FamilySurvived'] = round(all_data['Surname'].apply(lambda x: surname_mean[x]), 6)
    # 家庭成员分类：对家庭成员 >= 2 的乘客进行分组，分为妇女儿童组 / 成年男性组
    female_child_group = all_data[(all_data['FamilyGroup'] >= 2) & ((all_data['Age'] <= 12) | (all_data['Sex'] == 'female'))]
    male_adult_group = all_data[(all_data['FamilyGroup'] >= 2) & (all_data['Age'] > 12) & (all_data['Sex'] == 'male')]
    # 大部分平均存活率都为 1 或 0，要么全部幸存，要么全部遇难
    sns.countplot(x='FamilySurvived', data=female_child_group)
    sns.countplot(x='FamilySurvived', data=male_adult_group)
    # plt.show()

    # 不同家庭成员数量的乘客生还情况：新增 FamilySize 特征，并分为 3 类
    all_data['FamilySize'] = all_data['SibSp'] + all_data['Parch'] + 1
    all_data['FamilyLabel'] = all_data['FamilySize'].apply(family_label)
    sns.barplot(x='FamilyLabel', y='Survived', data=all_data)
    # plt.show()

    # 不同甲板的乘客生还情况：新增 Deck 特征，先将 Cabin 空缺值填充为 Unknown，再提取 Cabin 中的首字母构成甲板号
    all_data['Cabin'] = all_data['Cabin'].fillna('Unknown')
    all_data['Deck'] = all_data['Cabin'].str[0]
    sns.barplot(x='Deck', y='Survived', data=all_data)
    # plt.show()

    # 不同共票号数的乘客生还情况：新增 TicketGroup 特征，统计每个乘客的共票号数
    ticket_count = all_data['Ticket'].value_counts().to_dict()  # 将每个票号出现次数记为字典
    all_data['TicketCount'] = all_data['Ticket'].apply(lambda x: ticket_count[x])
    all_data['TicketCount'] = all_data['Ticket'].map(ticket_count)  # apply 和 map 两种写法都可以
    all_data['TicketGroup'] = all_data['TicketCount'].apply(ticket_label)
    sns.barplot(x='TicketGroup', y='Survived', data=all_data)
    # plt.show()
    return train, all_data


# 3、数据清洗
def s3_cleaning(train, all_data):
    # 1、缺失值填充
    # Age 缺失值较多，用 Sex, Title, Pclass 三个特征构建随机森林模型，填充年龄缺失值
    # 先将连续值处理为离散值，有大小意义的离散特征处理为数字
    age_df = all_data[['Age', 'Pclass', 'Sex', 'Title']]
    # 进行独热编码，实际上相当于 groupby 这四个字段的每一个值再 unstack，存在数据为 1，否则为 0
    age_df = pd.get_dummies(age_df)
    print(age_df.head(10))
    # as_maxtrix()：将值转换为数组（已经废弃，使用 values 代替）
    known_age = age_df[age_df['Age'].notnull()].values
    unknown_age = age_df[age_df['Age'].isnull()].values
    print(known_age)
    X = known_age[:, 1:]  # 自变量：全部行，第二列~最后一列（Sex, Title, Pclass）的离散值
    y = known_age[:, 0]  # 因变量：全部行，第一列（Age）
    # 随机森林模型参数：random_state 一个确定的随机值将会产生相同的结果，n_estimators 决策树个数（默认为100），n_jobs 处理器个数（-1 表示无限制）
    rfr = RandomForestRegressor(random_state=0, n_estimators=100, n_jobs=-1)
    # fit：使用已知 Sex, Title, Pclass 三个特征（X）和 Age（y）的数据训练随机森林模型
    rfr.fit(X, y)
    # predict：对年龄未知的数据进行预测（返回结果数组）；df[:, i:j:s]：全部行，i 到 j-1 列，步长为 s（step 默认为 1）
    predict_ages = rfr.predict((unknown_age[:, 1::]))
    # loc[row, col]：选出年龄为空的乘客的 Age 列，用预测值填充
    # PassengerId = 6 的乘客年龄未知，根据随机森林模型填充年龄为 28.75，也就是说，具有这三个相同特征的年龄未知的乘客，均填充为该年龄
    all_data.loc[all_data['Age'].isnull(), 'Age'] = predict_ages

    # 登船港口 Embarked 有 2 个缺失值，缺失乘客的客舱等级 Pclass 均为 1 且票价 Fare 皆为 80（这两个特征与登船港口关系最大）
    # print(data[data['Embarked'].isnull()])
    # Pclass 为 1，Embarked 为 C 的乘客，Fare 中位数接近 80，所以使用 C 填充 Embarked 为空的乘客
    print(all_data.groupby(by=['Pclass', 'Embarked'])['Fare'].aggregate(['median', 'mean']))
    all_data['Embarked'] = all_data['Embarked'].fillna('C')
    # print(data.loc[data['PassengerId'] == 62])

    # 票价 Fare 有 1 个缺失值，使用同登船港口 Embarked 和同票仓席位 Pclass 的乘客信息填充
    print(all_data[all_data['Fare'].isnull()])
    fare = all_data[(all_data['Embarked'] == 'S') & (all_data['Pclass'] == 3)]['Fare'].median()
    all_data['Fare'] = all_data['Fare'].fillna(fare)

    # 2、同组识别
    # 因为普遍规律是女性和儿童幸存率高，成年男性幸存率低，所以把不符合普遍规律的反常组选出来单独处理
    # 将女性和儿童组中幸存率为 0 的组设置为遇难组，将成年男性组中幸存率为 1 的组设置为幸存组
    # 推测遇难组中女性和儿童幸存的可能性比较低，幸存组中成年男性幸存的可能性比较高
    female_child_group = all_data[(all_data['FamilyGroup'] >= 2) & ((all_data['Age'] <= 12) | (all_data['Sex'] == 'female'))]
    male_adult_group = all_data[(all_data['FamilyGroup'] >= 2) & (all_data['Age'] > 12) & (all_data['Sex'] == 'male')]
    dead_list = set(female_child_group.loc[female_child_group['FamilySurvived'] == 0, 'Surname'])  # set 是不重复元素序列（相当于去重）
    survived_list = set(male_adult_group.loc[male_adult_group['FamilySurvived'] == 1, 'Surname'])  # to 是 pd 函数，set 是原生函数

    # 为了使这两组反常组中的样本能够被正确分类，对测试机中的反常组样本的 Age, Title, Sex 进行惩罚修改
    train = all_data[all_data['Survived'].notnull()]
    test = all_data[all_data['Survived'].isnull()]
    test.loc[(test['Surname'].apply(lambda x: x in dead_list)), 'Sex'] == 'male'
    test.loc[(test['Surname'].apply(lambda x: x in dead_list)), 'Age'] == 60
    test.loc[(test['Surname'].apply(lambda x: x in dead_list)), 'Title'] == 'Mr'
    test.loc[(test['Surname'].apply(lambda x: x in survived_list)), 'Sex'] == 'female'
    test.loc[(test['Surname'].apply(lambda x: x in survived_list)), 'Age'] == 5
    test.loc[(test['Surname'].apply(lambda x: x in survived_list)), 'Title'] == 'Miss'

    # 3、特征转换
    # 选取特征，转换为数值变量，划分训练集和测试集
    all_data = pd.concat([train, test], ignore_index=True)
    cols = ['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title', 'FamilyLabel', 'Deck', 'TicketGroup']
    all_data = all_data[cols]
    all_data = pd.get_dummies(all_data)
    train = all_data[all_data['Survived'].notnull()]
    test = all_data[all_data['Survived'].isnull()].drop('Survived', axis=1)
    return train, test


# 4、建模和优化
def s4_modeling(train, test, PassengerId):
    X = train.values[:, 1:]  # 自变量：全部行，第二列~最后一列
    y = train.values[:, 0]  # 因变量：全部行，第一列

    # 1、参数优化：用网格搜索自动选取最优参数
    pipe = Pipeline([('select', SelectKBest(k=20)),
                     ('classify', RandomForestClassifier(random_state=10, max_features='sqrt'))])
    param_test = {'classify__n_estimators': list(range(20, 50, 2)),
                  'classify__max_depth': list(range(3, 60, 3))}
    gsearch = GridSearchCV(estimator=pipe, param_grid=param_test, scoring='roc_auc', cv=10)
    gsearch.fit(X, y)
    print(gsearch.best_params_, gsearch.best_score_)

    # 2、训练模型
    select = SelectKBest(k=20)
    clf = RandomForestClassifier(random_state=10, warm_start=True, n_estimators=26, max_depth=6, max_features='sqrt')
    pipeline = make_pipeline(select, clf)
    pipeline.fit(X, y)

    # 3、交叉验证：将数据分成 10 折，每一折都当做一次测试集，其余 9 折当做训练集，循环 10 次，比较预测结果后计算模型评分
    cv_score = cross_val_score(pipeline, X, y, cv=10)
    print('CV Score: Mean - ', np.mean(cv_score), 'Std - ', np.std(cv_score))

    # 4、预测
    predictions = pipeline.predict(test)
    submission = pd.DataFrame({'PassengerId': PassengerId, 'Survived': predictions.astype(np.int32)})
    submission.to_csv('titanic/submission1.csv', index=False)


def family_label(s):
    if (s >= 2) & (s <= 4):
        return 2
    elif ((s > 4) & (s <= 7)) | (s == 1):
        return 1
    elif s > 7:
        return 0


def ticket_label(s):
    if (s >= 2) & (s <= 4):
        return 2
    elif ((s > 4) & (s <= 8)) | (s == 1):
        return 1
    elif s > 8:
        return 0


if __name__ == '__main__':
    set_option()
    train, test, passengerId, all_data = load_data()
    s1_preview(train)
    train, all_data = s2_preliminary(train, all_data)
    train, test = s3_cleaning(train, all_data)
    s4_modeling(train, test, passengerId)
