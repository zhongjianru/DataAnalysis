import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime

from matplotlib.font_manager import FontProperties


def log_of_aip(data, n, money):
    """
    data：测试数据集，即指数历史走势数据
    n：定投周期，6个月的整数倍，如n=1代表定投6个月，后面可以改成月
    money：定投金额
    """

    # 使用 [] 对 DataFrame 进行切片，按照索引进行列选择 ':' 全部数据 data.loc[<row selection>, <column selection>]
    data = data.loc[:, ['open', 'trade_date']]
    # DataFrame.apply(func, axis=0, broadcast=False, raw=False, reduce=None, args=(), **kwds)
    data['trade_month'] = data['trade_date'].apply(lambda x: str(x)[0:6])  # 匿名函数 lanbda x，取日期前六位

    # 根据指定的格式把一个时间字符串解析为时间元组
    # pandas 所有的 string 类型的 column 都为 object 类型
    data['date'] = data['trade_date'].apply(lambda x: datetime.strptime(x, '%Y%m%d'))
    data['date'] = pd.to_datetime(data['date'])  # 将 object 类型的日期转换成 datetime64 类型，就可以对日期进行取值了
    data['year'] = data['date'].map(lambda x: x.year)  # 这样获取年份比循环高效许多，datetime64 类型的日期可以这样获取年月
    # print(data)

    # DataFrame.set_index(keys, drop=True, append=False, inplace=False, verify_integrity=False) 设置索引
    # DataFrame.sort_index(axis=0, by=None, ascending=True) by 参数的作用是针对某一（些）列进行排序（不能对行使用 by 参数）
    data = data.set_index('date').sort_index()
    # print(data.loc[:'2015-01-10']) # 对索引进行排序，之后就可以进行范围切片了

    # 假设余额宝的年化收益率为 4%
    data['余额宝利率'] = (4.0 / 100 + 1) ** (1.0 / 250) - 1  # ** 乘方
    data['理财收益_净值'] = (data['余额宝利率'] + 1).cumprod()  # comprod 累乘，comsum 累加
    # print(data)

    # 选择每个月的第一个交易日进行定投
    # resample 函数可以进行重采样，这里是每月取出第一条记录，缺失值用前一天的数据代替（上月末），但是 date 是错的，显示的是月末日期，可以只看月份
    trading_day = data.resample('M', kind='date').first()
    # print(trading_day)

    # 确定循环次数，因为得保证满足定投周期
    # try:
    All_Sales = pd.DataFrame()
    for i in range(len(trading_day) - (6 * n)):
        # 在定投周期结束后一个月卖出
        trading_cycle = trading_day.iloc[i:i + 6 * n + 1]  # iloc 通过行号获取行数据，不能是字符，ix 既可以用行号也可以用行字符

        # 计算卖出点下个月的指数均值
        in_month = data[data['trade_month'] == list(trading_cycle['trade_month'][-1:])[0]]  # == list 相当于 in
        # 透视表 pivot_table 根据一个键或多个键做数据聚合，这里是根据 trade_month 计算 open 的分组平均值，values() 返回 dict 中的值
        sales_point = in_month.pivot_table(values='open', index='trade_month').mean().values[0]
        print(trading_cycle)

        # 定投指数基金
        AIP = pd.DataFrame(index=trading_cycle.index)  # 指定索引
        AIP['定投金额'] = int(money)

        # 以基金当天净值作为买入的价格
        AIP['基金价格'] = trading_cycle['open']
        AIP['购买基金份额'] = AIP['定投金额'] / AIP['基金价格']
        AIP['累计基金份额'] = AIP['购买基金份额'].cumsum()

        # 定投理财产品
        AIP['购买理财产品份额'] = AIP['定投金额'] / trading_cycle['理财收益_净值']
        AIP['累计理财产品份额'] = AIP['购买理财产品份额'].consum()  # 'Series' object has no attribute 'consum'
        print(AIP)

        # 计算每个交易日的本息（即本金+利息，公式=当天的份额✖当天的基金价格
        result = pd.concat([trading_cycle, AIP], axis=1)
        result['基金本息'] = (result['open'] * result['累计基金份额']).astype('int')
        result['理财本息'] = (result['理财收益_净值'] * result['累计理财产品份额']).astype('int')
        print('result:', result)  # 这里已经没有执行了，在这段之前已经返回了

        # 买入点 result['trade_date'][0]
        # 定投周期（月） 6*n
        # 定投投入本金 result['累计定投本金'][-2:-1][0]
        # 基金卖出后本息 result['累计基金份额'][-2:-1][0] * sales_point
        # 余额宝卖出后本息 result['理财本息'][-2:-1][0]

        # 这里没有考虑交易费，实际上权益类基金卖出的时候会收取赎回费，但是货币基金不需要

        Each_sales = pd.DataFrame([[result['trade_date'][0],
                                    6 * n,
                                    result['累计定投本金'][-2:-1][0],
                                    result['累计基金份额'][-2:-1][0] * sales_point,
                                    result['理财本息'][-2:-1][0]]],
                                    columns=['买入点', '定投周期（月）', '累计定投本金', '基金卖出后本息', '余额宝卖出后本息'])

        Each_sales['基金收益率%'] = 100 * (Each_sales['基金卖出后本息'][0] / Each_sales['累计定投本金'][0] - 1)
        Each_sales['余额宝收益率%'] = 100 * (Each_sales['余额宝卖出后本息'][0] / Each_sales['累计定投本金'][0] - 1)
        Each_sales['LikeOrNot'] = Each_sales['基金卖出后本息'] > Each_sales['余额宝卖出后本息']
        All_Sales = All_Sales.append(Each_sales)

    return All_Sales

# except:
#     print('定投周期大于历史股价走势！请重新设置定投周期。')


def Rate_of_Like(data, money):
    data = data
    money = money

    # 选择每个月的第一个交易日进行定投
    trading_day = data.resample('M', kind='date').first()  # Pandas 中的 resample 函数可以完成日期的聚合工作，这里是按月聚合

    Rate_of_Like = pd.DataFrame()
    for i in range(int(len(trading_day) / 6)):
        tt = log_of_aip(data_sh, i + 1, 2000)
        rate = pd.DataFrame([[(i + 1) * 6, (tt['LikeOrNot'].value_counts() / len(tt))[True]]],
                            columns=['定投周期(月)', '定投基金满意占比'])
        Rate_of_Like = Rate_of_Like.append(rate)
    return Rate_of_Like


if __name__ == '__main__':
    plt.rcParams['figure.figsize'] = (10.0, 4.0)  # 设置 figure_size 尺寸
    plt.rcParams['image.interpolation'] = 'nearest'  # 设置 interpolation style
    plt.rcParams['image.cmap'] = 'gray'  # 设置 颜色 style
    plt.rcParams['savefig.dpi'] = 100  # 图片像素
    plt.rcParams['figure.dpi'] = 100  # 分辨率

    # 利用 plotly 绘图
    import plotly.offline as of
    # import plotly.grid_objs as go
    import chart_studio.grid_objs as go

    # 设置 token
    ts.set_token('9717378f7eb2f18f1e47ca126d37c18aa8fcf0928688acd6c9f6a81f')

    # 初始化接口
    pro = ts.pro_api()

    # 获取基本信息
    sh_basic = pro.index_basic(market='SSE')  # 上交所
    sz_basic = pro.index_basic(market='SZSE')  # 深交所

    sh_basic['idx'] = sh_basic['ts_code'].apply(lambda x: str(x)[0:6])
    sh_basic = sh_basic.set_index('idx').sort_index()

    # print(type(sh_basic))
    # print(sh_basic)
    # print(sh_basic.loc[sh_basic['name'] == '上证综指']) # 掩码取值
    # print(sh_basic.loc['000001']) # 索引取值

    # 000001.SH 上证综指
    # 399001.SZ 深证成指

    data_sh = pro.index_daily(ts_code='000001.SH', start_date='20150101', end_date='20191231')
    data_sz = pro.index_daily(ts_code='399001.SZ', start_date='20150101', end_date='20191231')

    # 取消显示行列数的限制
    pd.options.display.max_columns = None
    pd.options.display.max_rows = None

    # print(data_sh.head(n=100)) # head 和 tail 可以指定条数
    # print(data_sh.describe()) # 统计信息
    # print(data_sh.columns) # 所有列
    # print(data_sh.dtypes) # 各列数据类型，trade_date object

    # 测试第一部分，现在提示 Empty DataFrame，定投周期n=2，data_sh有3个月数据
    test = log_of_aip(data_sh, 1, 1000)
    print(test)

    # 函数调用&猜想验证
    # rate_of_sh = Rate_of_Like(data_sh, 2000)
    # rate_of_sh.head(n=10)
    #
    # # 绘制定投满意概率图，offline 离线使用，保存在本地，init 初始化
    # of.offline.init_notebook_mode(connected=True)
    #
    # trace1 = go.Scatter(
    #     x=rate_of_sh['定投周期（月）'],
    #     y=rate_of_sh['定投基金满意占比'],
    #     mode='lines+markers',
    #     name='定投基金满意的次数占比'
    # )
    #
    # data = go.Data([trace1])
    #
    # layout = dict(title='是否存在最合适的定投周期？',
    #               yaxis=dict(showgrid=True,  # 网格
    #                          zeroline=False,  # 是否显示基线，即沿着(0,0)画出 x 轴和 y 轴
    #                          nticks=20,
    #                          showline=True,
    #                          title='定投基金满意的次数占比'
    #                          ),
    #               xaxis=dict(showgrid=True,  # 网格
    #                          zeroline=False,  # 是否显示基线，即沿着(0,0)画出 x 轴和 y 轴
    #                          nticks=20,
    #                          showline=True,
    #                          title='定投基金（月）'
    #                          )
    #               )
    #
    # fig = dict(data=data, layout=layout)
    # of.plot(fig, filename='rate_of_like')
