# Python机器学习：Sklearn 与 TensorFlow 机器学习实用指南
# 参考资料：https://github.com/ageron/handson-ml
# 中文翻译：https://github.com/it-ebooks/hands-on-ml-zh
# 第二章：加州房价模型

import os
import tarfile
import numpy as np
import pandas as pd
import hashlib
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from six.moves import urllib
import joblib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer as Imputer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, TransformerMixin

# 全局变量，大写，函数内变量小写
DOWNLOAD_ROOT = 'https://raw.githubusercontent.com/ageron/handson-ml/master/'
HOUSING_PATH = 'datasets/housing'
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + '/housing.tgz'


# 下载数据
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, 'housing.tgz')
    urllib.request.urlretrieve(housing_url, tgz_path)  # 下载文件
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)  # 解压
    housing_tgz.close()  # 如果使用with open结构，就不用close


# 加载数据
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, 'housing.csv')
    return pd.read_csv(csv_path)


# 整体数据情况
def check_housing_data(data):
    print(data.head())
    print(data.info())  # 可以看到总房间数有207个缺失值，除了离大海距离ocean_proximity是object之外，其余都是数值
    print(data['ocean_proximity'].value_counts())
    print(data.describe())  # 数值属性的概括，空值不计
    # 每一列都作为横坐标生成一个图，将横坐标分成50个柱，也就是50个区间（纵坐标代表区间内的个数）
    # 1、收入中位数貌似不是美元，数据经过缩放调整，过高会变成15，过低会变成5
    # 2、房屋年龄中位数housing_median_age和房屋价值中位数median_house_value也被设了上限，这可能是很严重的问题，因为后者是研究对象，机器学习算法可能学习到价格不会超出这个界限
    # 3、这些属性有不同的量度（特征缩放）
    # 4、许多柱状图的尾巴很长，相较于左边，他们在中位数的右边延伸过远，可以在后面尝试变换处理这些属性，使其符合正态分布
    housing.hist(bins=50, figsize=(20, 15))
    plt.show()


# 创建测试集
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))  # 对数据长度随机排列，作为index
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]  # 测试集index
    train_indices = shuffled_indices[test_set_size:]  # 训练集index
    return data.iloc[train_indices], data.iloc[test_indices]


def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio


# 如果再次运行程序，就会得到一个不一样的数据集，多次运行之后，就可能会得到整个数据集，有以下方法可以避免
# 1、保存第一次运行时得到的数据集，并在随后的过程中加载
# 2、在生成随机index之前，设置随机数生成器的种子（比如np.random.seed(42)，以产生总是相同的洗牌指数
# 但是一旦数据集更新，以上两个方法都会失效。所以通常的解决办法是使用每个实例的id来判断这个实例是否应该加入测试集（不是直接指定id）
def split_train_test_by_id(data, test_ratio, id_column='id', hash=hashlib.md5):
    data.reset_index()  # 不使用行索引作为id：保证新数据放在现有数据的尾部，且没有数据被删除
    data['id'] = data['longitude'] * 1000 + data['latitude']  # 使用最稳定的特征来创建id，在这里是经纬度
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]  # iloc使用数字，loc使用索引


# 上面的方法都是随机取样方法，这里使用分层取样（分层数不能过大，且每个分层要足够大）
# 收入中位数是预测房价中位数非常重要的属性，应该保证测试集可以代表整体数据集中的多种收入分类
# 因为收入中位数是连续的数值属性，所以要先创建收入类别属性
def split_train_test_by_slice(data, test_ratio):
    data['income_cat'] = np.ceil(data['median_income'] / 1.5)  # 将收入中位数除以1.5，用ceil对值进行舍入，以产生离散分类
    data['income_cat'].where(data['income_cat'] < 5, 5.0, inplace=True)  # 将所有大于5的分类归入到分类5
    # print(data['income_cat'].value_counts().sort_index()/len(data))  # 收入分类比例

    # 按照收入分类进行取样，split返回分组后的数据在原数组中的索引
    ss = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=42)
    for train_index, test_index in ss.split(data, data['income_cat']):
        strat_train_set = data.loc[train_index]
        strat_test_set = data.loc[test_index]

    for set in strat_train_set, strat_test_set:
        set.drop('income_cat', axis=1, inplace=True)  # 将数据复原
    return strat_train_set, strat_test_set


# 地理数据可视化
def view_housing_data(data):
    # 房屋分布散点图：将每个点的透明度设置小一点，可以更好的看出点的分布
    # 说明房价和位置（比如，靠海）和人口密度联系密切
    # 可以使用聚类算法来检测主要的聚集，用一个新的特征值测量聚集中心的距离。
    # 尽管北加州海岸区域的房价不是非常高，但离大海距离属性也可能很有用，所以这不是用一个简单的规则就可以定义的问题。
    # figure1 = plt.figure()
    data.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4,
              s=data['population']/100, label='population',  # 每个圈的半径表示街区的人口
              c=data['median_house_value'], cmap=plt.get_cmap('jet'), colorbar=True)  # 颜色代表价格

    # 查找关联：因为数据集并不是非常大，可以使用corr方法计算出每对属性间的标准相关系数（也称作皮尔逊相关系数）
    plt.figure()
    corr_matrix = data.corr()
    sns.heatmap(corr_matrix, vmax=1, square=True)
    print(corr_matrix['median_house_value'].sort_values(ascending=False))

    # 另一种检测属性间相关系数的方法：scatter_matrix方法，可以画出每个数值属性对每个其它数值属性的图，这里选择相关性最大的四个属性
    # 如果将每个变量对自己作图，主对角线（左上到右下）都会是直线图，所以展示的是每个属性的柱状图
    plt.figure()
    attributes = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']
    scatter_matrix(data[attributes], figsize=(12, 8))

    # 最有希望用来预测房价中位数的属性是收入中位数，因此将这张图放大，这张图说明了几点：
    # 1、相关性非常高；可以清晰地看到向上的趋势，并且数据点不是非常分散
    # 2、我们之前看到的最高价，清晰地呈现为一条位于 500000 美元的水平线。这张图也呈现了一些不是那么明显的直线：
    # 一条位于 450000 美元的直线，一条位于 350000 美元的直线，一条在 280000 美元的线，和一些更靠下的线。
    # 你可能希望去除对应的街区，以防止算法重复这些巧合。
    data.plot(kind='scatter', x='median_income', y='median_house_value', alpha=0.1)

    # 在上面几步，你发现了一些数据的巧合，需要在给算法提供数据之前，将其去除。
    # 你还发现了一些属性间有趣的关联，特别是目标属性。
    # 你还注意到一些属性具有长尾分布，因此你可能要将其进行转换（例如，计算其 log 对数）

    # 属性组合试验：尝试多种属性组合
    # 如果你不知道某个街区有多少户，该街区的总房间数就没什么用。你真正需要的是每户有几个房间。
    data['rooms_per_household'] = data['total_rooms'] / data['households']
    # 相似的，总卧室数也不重要：你可能需要将其与房间数进行比较。
    data['rooms_per_bedroom'] = data['total_rooms'] / data['total_bedrooms']
    # 每户的人口数也是一个有趣的属性组合。
    data['population_per_household'] = data['population'] / data['households']
    # 再看看相关矩阵：
    # 与总房间数或卧室数相比，每卧室房间数与房价中位数的关联性更强，每户的房间数也比街区的总房间数的关联性更强
    corr_matrix = data.corr()
    print(corr_matrix['median_house_value'].sort_values(ascending=False))

    plt.show()


rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):

    def __init__(self, add_bedrooms_per_room=True):  # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


# 自定义转换器：
# 通过选择对应的属性（数值或分类）、丢弃其它的，来转换数据，并将输出 DataFrame 转变成一个 NumPy 数组
class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values


# 数据清洗
# 参考：https://blog.csdn.net/weixin_41571493/article/details/82714759
def trans_housing_data(housing):
    # 6.0、将预测值和标签分开
    housing = housing.drop('median_house_value', axis=1)  # drop和fillna等函数中设置inplace=True在原数据上修改，不设置默认不修改，需要赋值给其他变量

    # 6.1、处理缺失值：1、去掉缺失的记录，2、去掉整个属性，3、进行赋值（0、平均值、中位数等）
    # housing.dropna(subset=['total_bedrooms'])  # 1
    # housing.drop('total_bedrooms', axis=1)  # 2
    # median = housing['total_bedrooms'].median()
    # housing['total_bedrooms'].fillna(median, inplace=True)  # 3

    # 自定义转换器：sklearn是鸭子类型的（而不是继承），需要依次执行fit、transform和fit_transform方法
    # sklearn提供Imputer类来处理缺失值
    housing_num = housing.drop('ocean_proximity', axis=1)  # 只有数值属性才有中位数
    imputer = Imputer(strategy='median')  # imputer.statistics_属性与housing_num.median().values结果一致
    imputer.fit(housing_num)  # 将imputer实例拟合到训练数据
    x = imputer.transform(housing_num)  # 将imputer应用到每个属性，结果为Numpy数组
    housing_tr = pd.DataFrame(x, columns=housing_num.columns)
    # print(housing_tr.describe())  # 缺失值补充完成，全部列都非空

    # 6.2、处理文本和类别属性：ocean_proximity
    housing_cat = housing['ocean_proximity']

    # 将文本属性转换为数值
    encoder = LabelEncoder()
    housing_cat_encoded = encoder.fit_transform(housing_cat)  # 译注：该转换器只能用来转换标签
    # housing_cat_encoded, housing_categories = housing_cat.factorize()
    # print(housing_cat_encoded[:20])  # 属性转换为数值完成
    # print(encoder.classes_)  # 属性映射

    # 这种做法的问题是，ML 算法会认为两个临近的值比两个疏远的值要更相似。显然这样不对（比如，分类 0 和 4 比 0 和 1 更相似）
    # 要解决这个问题，一个常见的方法是给每个分类创建一个二元属性：独热编码（One-Hot Encoding），只有一个属性会等于 1（热），其余会是 0（冷）
    encoder = OneHotEncoder()
    housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1, 1))  # reshape行转列，结果是一个稀疏矩阵，只存放非零元素的位置
    # print(housing_cat_1hot.toarray()[:10])  # 复原为密集数组

    # 将以上两步合并（LabelEncoder + OneHotEncoder）
    encoder = LabelBinarizer()
    housing_cat_1hot = encoder.fit_transform(housing_cat)
    # print(housing_cat_1hot)  # 默认返回密集数组，设置sparse_output=True可得到一个稀疏矩阵

    # 6.3、特征缩放：让所有的属性有相同的量度
    # 方法1：线性函数归一化（Min-Max scaling）：值被转变、重新缩放，直到范围变成 0 到 1，转换器MinMaxScaler
    # 方法2：标准化（standardization）：首先减去平均值（所以标准化值的平均值总是 0），然后除以方差，使得到的分布具有单位方差，转换器StandardScaler

    # 许多数据转换步骤，需要按一定的顺序执行，sklearn提供了类Pipeline（转换流水线）来完成
    # 调用流水线的fit方法，就会对所有转换器顺序调用fit_transform方法，将每次调用的输出作为参数传递给下一个调用，一直到最后一个估计器，它只执行fit方法
    num_attribs = list(housing_num)
    cat_attribs = ['ocean_proximity']

    # 处理缺失值
    num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', Imputer(strategy='median')),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler())
    ])

    # housing_num_tr = num_pipeline.fit_transform(housing_num)  # 单独运行流水线

    # 处理类别属性
    cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('label_binarizer', LabelBinarizer())
    ])

    full_pipeline = FeatureUnion(transformer_list=[
        ('num_pipeline', num_pipeline),
        ('cat_pipeline', cat_pipeline)
    ])

    # housing_prepared = full_pipeline.fit_transform(housing)  # 运行整个流水线，会报错，暂时不使用

    housing_prepared = housing_tr.copy()  # 建立副本，不要修改原来的数据
    housing_prepared['ocean_proximity'] = housing_cat_encoded  # 合并填充缺失值的结果+类别属性转换为数值的结果

    return housing_prepared


# 显示模型评分
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


# 选择并训练模型
def train_housing_data(housing):
    # 7.1、线性回归模型
    housing_labels = train_set['median_house_value'].copy()
    lin_reg = LinearRegression()
    lin_reg.fit(housing, housing_labels)  # 训练模型

    # 部分训练集验证
    some_data = housing.iloc[:5]
    some_labels = housing_labels[:5]
    # some_data = full_pipeline.transform(some_data)  # 先将训练集进行处理
    print(lin_reg.predict(some_data))
    print(some_labels)  # 将预测值与实际值进行对比

    # 用全部训练集来计算这个回归模型的RMSE
    housing_predicions = lin_reg.predict(housing)  # 数据拟合
    lin_mse = mean_squared_error(housing_labels, housing_predicions)  # 计算误差
    lin_rmse = np.square(lin_mse)
    print(lin_rmse)  # 线性回归模型的预测误差非常大

    # 7.2、决策树模型：可以发现数据中复杂的非线性关系
    tree_reg = DecisionTreeRegressor()
    tree_reg.fit(housing, housing_labels)  # 训练模型

    housing_predicions = tree_reg.predict(housing)
    tree_mse = mean_squared_error(housing_labels, housing_predicions)
    tree_rmse = np.square(tree_mse)
    print(tree_rmse)  # 决策树模型，没有误差？

    # 7.3、随机森林模型：通过用特征的随机子集训练许多决策树
    forest_reg = RandomForestRegressor()
    forest_reg.fit(housing, housing_labels)  # 训练模型

    housing_predicions = forest_reg.predict(housing)
    forest_mse = mean_squared_error(housing_labels, housing_predicions)
    forest_rmse = np.square(forest_mse)
    print(forest_rmse)

    # 使用交叉验证做更佳的评估
    # 随机地将训练集分成十个不同的子集，成为“折”，然后训练评估决策树模型 10 次，每次选一个不用的折来做评估，用其它 9 个来做训练
    # 结果是一个包含10 个评分的数组
    lin_scores = cross_val_score(lin_reg, housing, housing_labels, scoring='neg_mean_squared_error', cv=10)
    lin_rmse_scores = np.square(-lin_scores)
    display_scores(lin_rmse_scores)

    tree_scores = cross_val_score(tree_reg, housing, housing_labels, scoring='neg_mean_squared_error', cv=10)
    tree_rmse_scores = np.square(-tree_scores)
    display_scores(tree_rmse_scores)  # 决策树模型过拟合很严重，它的性能比线性回归模型还差

    forest_scores = cross_val_score(tree_reg, housing, housing_labels, scoring='neg_mean_squared_error', cv=10)
    forest_rmse_scores = np.square(-forest_scores)
    display_scores(forest_rmse_scores)  # 现在好多了：随机森林看起来很有希望

    # 保存试验过的模型
    # joblib.dump(my_model, "my_model.pkl")
    # my_model_loaded = joblib.load("my_model.pkl")


if __name__ == '__main__':

    pd.set_option('display.max_columns', 20)  # 最大列数
    pd.set_option('display.width', 1000)  # 总宽度（字符数）

    # 1、获取数据
    # fetch_housing_data()

    # 2、加载数据
    housing = load_housing_data()

    # 3、整体数据情况
    # check_housing_data(housing)

    # 4、拆分数据集（很早就要进行，如果提前查看了测试集，可能会发生过拟合的情况）
    # train_set, test_set = split_train_test(housing, 0.2)  # 使用index进行采样，每次的测试集都不一样
    # train_set, test_set = split_train_test_by_id(housing, 0.2, 'id')  # 构造id，使用id进行采样
    # train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)  # sklearn分割数据集，random_state生成器种子
    train_set, test_set = split_train_test_by_slice(housing, 0.2)  # 分层取样

    # 对训练集进行研究
    housing = train_set.copy()

    # 5、地理数据可视化
    # view_housing_data(housing)

    # 6、数据清洗
    housing = trans_housing_data(housing)

    # 7、选择并训练模型
    train_housing_data(housing)

