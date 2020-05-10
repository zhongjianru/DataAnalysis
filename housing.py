# Python机器学习：Sklearn 与 TensorFlow 机器学习实用指南
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
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

# 全局变量，大写，函数内变量小写
DOWNLOAD_ROOT = 'https://raw.githubusercontent.com/ageron/handson-ml/master/'
HOUSING_PATH = 'datasets/housing'
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + '/housing.tgz'


# 下载数据
def fetching_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
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


if __name__ == '__main__':

    pd.set_option('display.max_columns', 20)  # 最大列数
    pd.set_option('display.width', 1000)  # 总宽度（字符数）

    # 1、获取数据
    # fetching_housing_data()

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
    view_housing_data(housing)

    # 6、为机器学习算法准备数据 p60




