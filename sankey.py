import pandas as pd
from pyecharts.charts import Sankey
from pyecharts import options as opts

# df1 = pd.read_excel('石油产量排名.xlsx')
# df1.head()

state = ['加拿大', '墨西哥', '美国', '阿根廷', '巴西']
continent = ['北美洲', '北美洲', '北美洲', '南美洲', '南美洲']
num = [255.5, 102.3, 669.4, 27.6, 140.3]

df1 = pd.DataFrame({'state': state, 'continent': continent, 'num': num})
print(df1)

# 产生节点
nodes = []

for i in set(pd.concat([df1.state, df1.continent])):
    dic_ = {}
    dic_['name'] = i
    nodes.append(dic_)

# 产生链接
links = []

# zip: 将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表
# a = [1,2,3]
# b = [4,5,6]
# zip(a,b)
# [(1, 4), (2, 5), (3, 6)]
for x, y, z in zip(df1.state, df1.continent, df1.num):
    dic_ = {}
    dic_['source'] = x
    dic_['target'] = y
    dic_['value'] = z
    links.append(dic_)

    colors = ['#54B4F9', '#F29150', '#FF7BAE', '#D69AC0', '#485CE0', '#28BE7A']

    s = Sankey(init_opts=opts.InitOpts(width='1350px', height='1350px'))
    s.set_colors(colors)
    s.add('sankey',
          nodes,
          links,
          pos_left='10%',
          pos_right='60%',
          linestyle_opt=opts.LineStyleOpts(opacity=0.2, curve=0.5, color='source'),
          itemstyle_opts=opts.ItemStyleOpts(border_width=1, border_color="#aaa"),
          tooltip_opts=opts.TooltipOpts(trigger_on="mousemove"),
          is_draggable=True,
          label_opts=opts.LabelOpts(position="left",
                                    font_family='Arial',
                                    margin=10,
                                    font_size=13,
                                    font_style='italic')
          )

    s.set_global_opts(title_opts=opts.TitleOpts(title='世界石油产量top30国家分布'))
    s.render('桑基图.html')
