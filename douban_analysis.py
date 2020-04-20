import pandas as pd
import numpy as np
import re
import jieba.analyse

from pyecharts import options as opts
from pyecharts.globals import SymbolType
from pyecharts.charts import Pie
from pyecharts.charts import Line
from pyecharts.charts import Bar
from pyecharts.charts import Map
from pyecharts.charts import WordCloud


def trans_all_cmts():
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_columns', 1000)
    pd.set_option('display.max_colwidth', 20)
    columns = ['cid', 'type', 'score', 'attitude', 'date', 'username', 'comment', 'votes', 'userpage', 'location']

    # 1、读取excel数据，读取进来的df中出了数字是int64类型的，其余全是ojbect类型，不是str，不能直接比较
    df_comments = df = pd.read_excel('files/comments.xls', usecols='B:K')  # excel读取范围
    # python语法糖：a < x < b, a = x = b, condition and a or b(c为真则a否则b), value if a else b
    # 读取多列df要传入一个list，df[['cid', 'type']]相当于df[columns]，或者df.iloc[[:, 1,2]]
    # df基础属性：shape, dtypes, ndim, index, columns, values->(索引值为index.values)
    # df整体情况：head(), tail(), info(), describe()
    # df定位：df.loc[0], df.iloc('idx'), df[df['col']=='value']
    # df合并：横向拼接列merge(df1, df2), 纵向拼接行df1.append(df2)和concat(df1, df2)
    # df方法：python方法是list(str), df方法是df[col].tolist(), 是后置的，便于链式调用

    # 2、处理空值
    # 2.0、删除空行
    df.dropna()
    # 2.1、筛选包含空值的行（不加转置T的话就是筛选包含空值的列），isnull返回布尔值，any筛选为True的值
    # print(df[df.isnull().T.any()])
    # 2.2、填充空值
    fill_na = {'location': '未知'}
    df.fillna(value=fill_na, inplace=True)  # df的空值填充了，但是赋值给另一个df是none
    # 2.3、将国外的地址统一转换成国外
    # df['location'][-1:]取最后一条记录，df['location'].apply(lambda x: x[-1:])取最后一个字符，判断最后一个字符是否为中文
    f = lambda x: '国外' if not ('\u4e00' <= x[-1:] <= '\u9fff') else x  # lambda可以简化表达式
    df['location'] = df['location'].apply(f)  # 这里map和apply都可以
    # 2.4、剔除空值和国外地址
    # 可以删除行或列，axis=0默认删除行，inplace为True直接在原数据上操作，为False返回一个新的df
    df_drop = df[df['location'].isin(['国外', '未知'])]  # 不想重复写这一列，可以用df.isin(list)
    df.drop(index=df_drop.index, inplace=True)  # 使用上面筛选出的index来删除行
    # 2.5、将省市分开
    df_province = pd.read_excel('files/province.xls')
    # df['province'] =

    # 3、处理日期，将日期拆分成年月日和时间
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].apply(lambda x: x.year)  # 获取年份，直接df['date'].year不行，要用apply或者map
    df['dt_date'] = df['date'].dt.date  # 获取日期

    # 4、评分转换：定义字典dict，df[col].map(dict)
    rating_dict = {'很差': '1星', '较差': '2星', '还行': '3星', '推荐': '4星', '力荐': '5星'}
    df['star'] = df['attitude'].map(rating_dict)
    # print(df.head(5))

    # 5、写出各类聚合数据，并存入df中
    return df


def show_pie_charts(df):
    score_perc = df['star'].value_counts() / df['star'].value_counts().sum()  # 评分占比 = 评分1-5的个数(分别) / 评分总个数
    score_perc = np.round(score_perc * 100, 2)
    score_perc = score_perc.sort_index()  # Series也可以排序
    # 聚合后类型为Series，转成Dataframe
    # df_score = pd.DataFrame(list(zip(score_perc.index, score_perc.values)), columns=['score', 'percentage']).sort_values('score')

    pie1 = Pie(init_opts=opts.InitOpts(width='1350px', height='750px'))
    pie1.add(series_name='评分占比',
             data_pair=[*zip(score_perc.index, score_perc.values)],  # Series: zip打包为一个元组对象，用*解压成元组，再[]转成列表
             # data_pair=[*zip(df_score['score'], df_score['percentage'])],  # Dataframe
             label_opts=opts.LabelOpts(formatter='{c}%'),  # formatter='{c}%'百分比
             radius=['35%', '70%']  # 是什么意思
             )
    pie1.set_global_opts(title_opts=opts.TitleOpts(title='总体评分分布'),
                         legend_opts=opts.LegendOpts(orient='vertical', pos_top='15%', pos_left='2%'),  # 图例的位置：水平/垂直
                         toolbox_opts=opts.ToolboxOpts())  # 右上角的工具栏，保存图片等等
    pie1.set_colors(['#D7655A', '#FFAF34', '#3B7BA9', '#EF9050', '#6FB27C'])
    pie1.render('charts/pie_chart.html')


def show_line_charts(df):
    date_num = df['dt_date'].value_counts().sort_index()  # 相当于count(1) group by date order by date
    line1 = Line(init_opts=opts.InitOpts(width='1350px', height='750px'))
    line1.add_xaxis(date_num.index.tolist())
    line1.add_yaxis(series_name='评论热度',
                    y_axis=date_num.values.tolist(),
                    areastyle_opts=opts.AreaStyleOpts(opacity=0.5),
                    label_opts=opts.LabelOpts(is_show=False))
    line1.set_global_opts(title_opts=opts.TitleOpts(title='评论热度时间走势图'),
                          toolbox_opts=opts.ToolboxOpts(),
                          visualmap_opts=opts.VisualMapOpts(max_=12))  # 定义最高热度的值
    line1.render('charts/line_chart.html')
    # print(date_num)


def show_bar_charts(df):
    city_top10 = df['location'].value_counts()[:10]  # 如果清洗出城市可以把这里改成城市
    bar1 = Bar(init_opts=opts.InitOpts(width='1350px', height='750px'))
    bar1.add_xaxis(city_top10.index.tolist())
    bar1.add_yaxis(series_name='城市评论', yaxis_data=city_top10.values.tolist())
    bar1.set_global_opts(title_opts=opts.TitleOpts(title='评论用户城市前十分布'),
                         toolbox_opts=opts.ToolboxOpts(),
                         visualmap_opts=opts.VisualMapOpts(max_=80))  # 定义城市评论最大值，最大值图例颜色为红色（热力）
    bar1.render('charts/bar_chart.html')
    # print(city_top10)


def show_map_charts(df):
    # 计数，[0:10]取前10
    city_num = df['location'].value_counts()
    # 返回index列表和value列表组成的元组[('北京', 62), ('上海', 50)]，再将元组转换成列表[['北京', 62], ['上海', 50]]
    city_list = [list(z) for z in zip(city_num.index.tolist(), city_num.values.tolist())]
    print(city_list)
    map1 = Map(init_opts=opts.InitOpts(width='1350px', height='750px'))
    map1.add(series_name='', data_pair=city_list, maptype='china')
    map1.set_global_opts(title_opts=opts.TitleOpts(title='评论用户国内城市分布'),
                         visualmap_opts=opts.VisualMapOpts(max_=50),
                         toolbox_opts=opts.ToolboxOpts())
    map1.render('charts/map_chart.html')  # 要先把省份去掉再显示，province=[]再replace


def show_word_charts(df):
    # 添加关键词，不会被分词掉
    key_words = ['男主', '女主', '郑京浩', '金素妍']
    for i in key_words:
        jieba.add_word(i)

    # 对用户评论进行分词
    text = df['comment'].str.cat(sep='。')  # 使用句号拼接评论
    word_num = jieba.analyse.extract_tags(text, topK=100, withWeight=True, allowPOS=())  # 权重前100个词，list里嵌套tuple

    # 停用词：在信息处理过程中过滤掉的某些字或词
    stop_words = []
    with open('files/stopwords.txt', mode='r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            stop_words.append(line.strip())

    stop_words_extend = ['韩剧', '男二', '真的', '还是', '这部', '完全', '一部', '撑起', '最后', '就算'
                         '但是', '陷入', '不错', '觉得', '这么', '简直', '爱情', '男女', '实在', '那么',
                         '一集', '虽然', '郑敬', '各种', '爱上', '这个', '整部', '时候', '看过', '有点',
                         '居然', '不要', '评分', '主角', '素妍', '现在', '果然', '怎么', '部分', '在于',
                         '因为', '...', 'bug', '30']
    stop_words.extend(stop_words_extend)  # 单个元素用append，list用extend

    # 去除停用词
    word_num_selected = []
    for i in word_num:
        if i[0] not in stop_words:
            word_num_selected.append(i)

    # 绘制词云图
    df_keywords = pd.DataFrame(word_num_selected, columns=['word', 'num'])
    word1 = WordCloud(init_opts=opts.InitOpts(width='950px', height='750px'))
    word1.add(series_name='关键词',
              data_pair=[*zip(df_keywords['word'], df_keywords['num'])],
              shape=SymbolType.ARROW,
              pos_top='0%',
              pos_left='0%')
    word1.set_global_opts(title_opts=opts.TitleOpts('用户评论词云图'),
                          toolbox_opts=opts.ToolboxOpts())
    word1.render('charts/word_chart.html')
    # print(df_keywords)


if __name__ == '__main__':
    df_comments = trans_all_cmts()
    show_pie_charts(df_comments)
    show_line_charts(df_comments)
    show_bar_charts(df_comments)
    show_word_charts(df_comments)
    show_map_charts(df_comments)
