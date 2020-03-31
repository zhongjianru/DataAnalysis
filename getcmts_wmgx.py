# 导入所需包
import pandas as pd
import requests
import parsel
import re
import time
from fake_useragent import UserAgent
import jieba

# 折线图
from pyecharts.charts import Line

# 饼图
from pyecharts.charts import Pie
from pyecharts import options as opts

# 条形图
from pyecharts.charts import Bar

# 地图
from pyecharts.charts import Map

# 词云图
from pyecharts.charts import WordCloud
from pyecharts.globals import SymbolType


# 登录豆瓣账号
def login_douban():
    """
    功能：登录豆瓣，维持会话形式
    """
    global s
    # 初始化session
    s = requests.Session()

    # 登录地址
    login_url = 'https://accounts.douban.com/j/mobile/login/basic'
    # 添加headers
    headers = {'user-agent': UserAgent().random}
    # 表单数据
    form_data = {
        'name': '你的账号',
        'password': '你的密码',
        'remember': 'false'
    }

    # post登录
    try:
        s.post(login_url, headers=headers, data=form_data)
    except:
        print("登录失败")


# 获取豆瓣短评
def get_one_page(url):
    """
    功能：给定URL地址，获取豆瓣电影一页的短评信息
    :param url: 电影URL地址
    :return: 返回数据框
    """

    # 添加headers
    headers = {'user-agent': UserAgent().random}

    # 发起请求
    try:
        r = s.get(url, headers=headers, timeout=5)
    except:
        time.sleep(3)
        r = s.get(url, headers=headers, timeout=5)

    # 解析网页
    data = parsel.Selector(r.text)

    # 获取用户名
    user_name = [re.findall(r'.*?class="">(.*?)</a>.*', i)
                 for i in data.xpath('//span[@class="comment-info"]').extract()]
    # 获取评分
    rating = [re.findall(r'.*?<span class="allstar\d+ rating" title=(.*?)></span>.*', i)
              for i in data.xpath('//span[@class="comment-info"]').extract()]
    # 获取评论时间
    comment_time = [re.findall(r'<span class="comment-time " title=(.*)>.*', i)
                    for i in data.xpath('//span[@class="comment-info"]').extract()]
    # 获取短评信息
    comment_info = data.xpath('//span[@class="short"]/text()').extract()
    # 投票次数
    votes_num = data.xpath('//span[@class="comment-vote"]/span/text()').extract()
    # 获取主页URL
    user_url = data.xpath('//div[@class="avatar"]/a/@href').extract()

    # 保存数据
    df_one = pd.DataFrame({
        'user_name': user_name,
        'rating': rating,
        'comment_time': comment_time,
        'comment_info': comment_info,
        'votes_num': votes_num,
        'user_url': user_url
    })

    return df_one


# 获取短评用户信息
def get_all_pages(movie_id, page_num=25):
    """
    功能：获取豆瓣电影25页短评信息
    :param movie_id: 电影ID
    :param page_num: 爬取页面数
    :return: 返回数据框
    """
    df_25 = pd.DataFrame()

    for i in range(page_num):
        # 构造URL
        url = 'https://movie.douban.com/subject/{}/comments?start={}&limit=20&sort=new_score&status=P'.format(movie_id,
                                                                                                              i * 20)
        # 调用函数
        df = get_one_page(url)
        # 循环追加
        df_25 = df_25.append(df, ignore_index=True)
        # 打印进度
        print('我正在获取第{}页的信息'.format(i + 1))
        # 休眠一秒
        time.sleep(1)
    return df_25


# 转换评论信息
def trans_all_cmts():
    """
    推荐星级：转换为1~5分
    评论时间：转换为时间类型，并提取日期数据
    城市信息：有未填写数据、海外城市、写错的需要进行处理
    短评信息：需要进行分词处理
    """
    # 处理评分列
    df['rating'] = [re.sub(r'\[\'\"|\"\'\]', '', i) for i in df['rating']]

    # 替换空列表
    df['rating'].replace('[]', '还行', inplace=True)

    # 定义字典
    rating_dict = {
        '很差': '1星',
        '较差': '2星',
        '还行': '3星',
        '推荐': '4星',
        '力荐': '5星'
    }

    df['rating'] = df['rating'].map(rating_dict)

    # 评论信息分词处理
    # 合并为一篇
    txt = df['comment_info'].str.cat(sep='。')

    # 添加关键词
    jieba.add_word('黄轩')
    jieba.add_word('佟丽娅')
    jieba.add_word('男主')
    jieba.add_word('女主')
    jieba.add_word('跳戏')
    jieba.add_word('颜值')
    jieba.add_word('吐槽')
    jieba.add_word('装逼')
    jieba.add_word('国产剧')

    # 读入停用词表
    stop_words = []
    with open('stop_words.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            stop_words.append(line.strip())

    # 添加停用词
    stop_words.extend(['一部', '一拳', '一行', '10', '啊啊啊', '一句',
                       'get', '哈哈哈哈', '哈哈哈', '越来越', '一步',
                       '一种', '样子', '几个', '第一集', '一点',
                       '第一', '没见', '一集', '第一次', '两个',
                       '二代', '真的', '2020', '令人'])

    # 评论字段分词处理
    word_num = jieba.analyse.extract_tags(txt,
                                          topK=100,
                                          withWeight=True,
                                          allowPOS=())

    # 去停用词
    word_num_selected = []

    for i in word_num:
        if i[0] not in stop_words:
            word_num_selected.append(i)

    key_words = pd.DataFrame(word_num_selected, columns=['words', 'num'])


# 绘制饼图
def show_pie_charts():
    score_perc = df['rating'].value_counts() / df['rating'].value_counts().sum()
    score_perc = np.round(score_perc * 100, 2)
    print(score_perc)

    pie1 = Pie(init_opts=opts.InitOpts(width='1350px', height='750px'))
    pie1.add("",
             [*zip(score_perc.index, score_perc.values)],
             radius=["35%", "70%"])
    pie1.set_global_opts(title_opts=opts.TitleOpts(title='总体评分分布'),
                         legend_opts=opts.LegendOpts(orient="vertical", pos_top="15%", pos_left="2%"),
                         toolbox_opts=opts.ToolboxOpts())
    pie1.set_series_opts(label_opts=opts.LabelOpts(formatter="{c}%"))
    pie1.set_colors(['#D7655A', '#FFAF34', '#3B7BA9', '#EF9050', '#6FB27C'])
    pie1.render()


# 绘制折线图
def show_line_charts():
    df['comment_time'] = pd.to_datetime(df['comment_time'])
    df['comment_date'] = df['comment_time'].dt.date
    comment_num = df['comment_date'].value_counts().sort_index()

    line1 = Line(init_opts=opts.InitOpts(width='1350px', height='750px'))
    line1.add_xaxis(comment_num.index.tolist())
    line1.add_yaxis('评论热度', comment_num.values.tolist(),
                    areastyle_opts=opts.AreaStyleOpts(opacity=0.5),
                    label_opts=opts.LabelOpts(is_show=False))
    line1.set_global_opts(title_opts=opts.TitleOpts(title='时间走势图'),
                          toolbox_opts=opts.ToolboxOpts(),
                          visualmap_opts=opts.VisualMapOpts(max_=200))
    line1.render()


# 绘制条形图
def show_bar_charts():
    # 国内城市top10
    city_top10 = df['city_dealed'].value_counts()[:12]
    city_top10.drop('国外', inplace=True)
    city_top10.drop('未填写', inplace=True)

    bar1 = Bar(init_opts=opts.InitOpts(width='1350px', height='750px'))
    bar1.add_xaxis(city_top10.index.tolist())
    bar1.add_yaxis("城市", city_top10.values.tolist())
    bar1.set_global_opts(title_opts=opts.TitleOpts(title="评论者Top10城市分布"),
                         visualmap_opts=opts.VisualMapOpts(max_=50),
                         toolbox_opts=opts.ToolboxOpts())
    bar1.render()


# 绘制地图分布
def show_map_charts():
    city_num = df['city_dealed'].value_counts()
    city_num.drop('国外', inplace=True)
    city_num.drop('未填写', inplace=True)

    map1 = Map(init_opts=opts.InitOpts(width='1350px', height='750px'))
    map1.add("", [list(z) for z in zip(city_num.index.tolist(), city_num.values.tolist())],
             maptype='china')
    map1.set_global_opts(title_opts=opts.TitleOpts(title='评论者国内城市分布'),
                         visualmap_opts=opts.VisualMapOpts(max_=50),
                         toolbox_opts=opts.ToolboxOpts())
    map1.render()


# 绘制词云图
def show_word_charts():
    word1 = WordCloud(init_opts=opts.InitOpts(width='1350px', height='750px'))
    word1.add("", [*zip(key_words.words, key_words.num)],
              word_size_range=[20, 200],
              shape=SymbolType.DIAMOND)
    word1.set_global_opts(title_opts=opts.TitleOpts('完美关系豆瓣短评词云图'),
                          toolbox_opts=opts.ToolboxOpts())
    word1.render()


if __name__ == '__main__':
    # 先登录豆瓣
    login_douban()
    # 获取完美关系
    df_all = get_all_pages(movie_id='30221758')
    print(df_all.shape)
