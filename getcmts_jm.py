# 导入所需包
import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options


# 登录豆瓣
# 为了解决登录的问题，本次使用Selenium+BeautifulSoup获取数据
def login_douban():
    global browser  # 设置为全局变量
    browser = webdriver.Chrome()

    # 进入登录页面
    login_url = 'https://accounts.douban.com/passport/login?source=movie'
    browser.get(login_url)

    # 点击密码登录
    browser.find_element_by_class_name('account-tab-account').click()

    # 输入账号和密码
    username = browser.find_element_by_id('username')
    username.send_keys('18511302788')
    password = browser.find_element_by_id('password')
    password.send_keys('12349148feng')

    # 点击登录
    browser.find_element_by_class_name('btn-account').click()


# 定义函数获取单页数据
def get_one_page(url):
    '''功能：传入url，豆瓣电影一页的短评信息'''
    # 进入短评页
    browser.get(url)

    # 使用bs解析网页数据
    bs = BeautifulSoup(browser.page_source, 'lxml')

    # 获取用户名
    username = [i.find('a').text for i in bs.findAll('span', class_='comment-info')]
    # 获取用户url
    user_url = [i.find('a')['href'] for i in bs.findAll('span', class_='comment-info')]

    # 获取推荐星级
    rating = []
    for i in bs.findAll('span', class_='comment-info'):
        try:
            one_rating = i.find('span', class_='rating')['title']
            rating.append(one_rating)
        except:
            rating.append('力荐')

    # 评论时间
    time = [i.find('span', class_='comment-time')['title'] for i in bs.findAll('span', class_='comment-info')]
    # 短评信息
    short = [i.text for i in bs.findAll('span', class_='short')]
    # 投票次数
    votes = [i.text for i in bs.findAll('span', class_='votes')]

    # 创建一个空的DataFrame
    df_one = pd.DataFrame()
    # 存储信息
    df_one['用户名'] = username
    df_one['用户主页'] = user_url
    df_one['推荐星级'] = rating
    df_one['评论时间'] = time
    df_one['短评信息'] = short
    df_one['投票次数'] = votes

    # 用户主页的地址可以获取到用户的城市信息，这一步比较简单，此处的代码省略

    return df_one


# 传入电影ID，获取豆瓣电影25页的短评信息
def get_25_page(movie_id):
    # 创建空的DataFrame
    df_all = pd.DataFrame()

    # 循环追加
    for i in range(25):
        url = "https://movie.douban.com/subject/{}/comments?start={}&limit=20&sort=new_score&status=P".format(movie_id,i*20)
        print('我正在抓取第{}页'.format(i+1), end='\r')
        # 调用函数
        df_one = get_one_page(url)
        df_all = df_all.append(df_one, ignore_index=True)
        # 程序休眠一秒
        time.sleep(1.5)
    return df_all


# 推荐星级：转换为1-5分。
# 评论时间：转换为时间类型，提取出日期信息
# 城市：有城市空缺、海外城市、乱写和pyecharts尚4. 不支持的城市，需要进行处理
# 短评信息：需要进行分词和提取关键词

# 定义函数转换推荐星级字段
def transform_star(x):
    if x == '力荐':
        return 5
    elif x == '推荐':
        return 4
    elif x == '还行':
        return 3
    elif x == '较差':
        return 2
    else:
        return 1

# 星级转换
df_all['星级'] = df_all.推荐星级.map(lambda x:transform_star(x))
# 转换日期类型
df_all['评论时间'] = pd.to_datetime(df_all.评论时间)
# 提取日期
df_all['日期'] = df_all.评论时间.dt.date

# 定义函数-获取短评信息关键词
def get_comment_word(df):
    '''功能：传入df,提取短评信息关键词'''
    # 导入库
    import jieba.analyse
    import os
    # 去停用词
    stop_words = set()

    # 加载停用词
    cwd = os.getcwd()
    stop_words_path = cwd + '\\stop_words.txt'

    with open(stop_words_path, 'r', encoding='utf-8') as sw:
         for line in sw.readlines():
            stop_words.add(line.strip())

    # 添加停用词
    stop_words.add('6.3')
    stop_words.add('一张')
    stop_words.add('一部')
    stop_words.add('徐峥')
    stop_words.add('徐导')
    stop_words.add('电影')
    stop_words.add('电影票')

    # 合并评论信息
    df_comment_all = df['短评信息'].str.cat()

    # 使用TF-IDF算法提取关键词
    word_num = jieba.analyse.extract_tags(df_comment_all, topK=100, withWeight=True, allowPOS=())

    # 做一步筛选
    word_num_selected = []

    # 筛选掉停用词
    for i in word_num:
        if i[0] not in stop_words:
            word_num_selected.append(i)
        else:
            pass

    return word_num_selected

key_words = get_comment_word(df_all)
key_words = pd.DataFrame(key_words, columns=['words','num'])


# 总体评分百分比
score_perc = df_all.星级.value_counts() / df_all.星级.value_counts().sum()
score_perc = np.round(score_perc*100,2)

# 导入所需包
from pyecharts import options as opts
from pyecharts.charts import Pie, Page

# 绘制柱形图
pie1 = Pie(init_opts=opts.InitOpts(width='1350px', height='750px'))
pie1.add("",
         [*zip(score_perc.index, score_perc.values)],
         radius=["40%","75%"])
pie1.set_global_opts(title_opts=opts.TitleOpts(title='总体评分分布'),
                     legend_opts=opts.LegendOpts(orient="vertical", pos_top="15%", pos_left="2%"),
                     toolbox_opts=opts.ToolboxOpts())
pie1.set_series_opts(label_opts=opts.LabelOpts(formatter="{c}%"))
pie1.render('总体评分分布.html')


# 时间排序
time = df_all.日期.value_counts()
time.sort_index(inplace=True)

from pyecharts.charts import Line

# 绘制时间走势图
line1 = Line(init_opts=opts.InitOpts(width='1350px', height='750px'))
line1.add_xaxis(time.index.tolist())
line1.add_yaxis('评论热度', time.values.tolist(), areastyle_opts=opts.AreaStyleOpts(opacity=0.5), label_opts=opts.LabelOpts(is_show=False))
line1.set_global_opts(title_opts=opts.TitleOpts(title="时间走势图"), toolbox_opts=opts.ToolboxOpts())
line1.render('评论时间走势图.html')


# 国内城市top10
city_top10 = df_all.城市处理.value_counts()[:12]
city_top10.drop('国外', inplace=True)
city_top10.drop('未知', inplace=True)

from pyecharts.charts import Bar

# 条形图
bar1 = Bar(init_opts=opts.InitOpts(width='1350px', height='750px'))
bar1.add_xaxis(city_top10.index.tolist())
bar1.add_yaxis("城市", city_top10.values.tolist())
bar1.set_global_opts(title_opts=opts.TitleOpts(title="评论者Top10城市分布"),toolbox_opts=opts.ToolboxOpts())
bar1.render('评论者Top10城市分布条形图.html')


city_num = df_all.城市处理.value_counts()[:30]
city_num.drop('国外', inplace=True)
city_num.drop('未知', inplace=True)

c1 = Geo(init_opts=opts.InitOpts(width='1350px', height='750px'))
c1.add_schema(maptype='china')
c1.add('geo', [list(z) for z in zip(city_num.index, city_num.values.astype('str'))], type_=ChartType.EFFECT_SCATTER)
c1.set_series_opts(label_opts=opts.LabelOpts(is_show=False))
c1.set_global_opts(visualmap_opts=opts.VisualMapOpts(),
                   title_opts=opts.TitleOpts(title='评论者城市分布'),
                   toolbox_opts=opts.ToolboxOpts())
c1.render('评论者城市分布地图.html')


from pyecharts.charts import WordCloud
from pyecharts.globals import SymbolType, ThemeType

word = WordCloud(init_opts=opts.InitOpts(width='1350px', height='750px'))
word.add("", [*zip(key_words.words, key_words.num)], word_size_range=[20, 200])
word.set_global_opts(title_opts=opts.TitleOpts(title="囧妈电影评论词云图"),
                    toolbox_opts=opts.ToolboxOpts())
word.render('囧妈电影评论词云图.html')


if __name__ == '__main__':
    # 先运行登录函数
    login_douban()
    # 程序休眠两秒
    time.sleep(2)
    # 再运行循环翻页函数
    movie_id = 30306570  # 囧妈
    df_all = get_25_page(movie_id)