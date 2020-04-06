import jieba
import pandas as pd


# 1、通过键值对的形式存储词语及其出现的次数
def show_txt_words_f1():
    txt = open("C:\\Users\\Zero\\Desktop\\tools\\Python数据科学手册_机器学习.md", "r", encoding='utf-8').read()
    words = jieba.lcut(txt)  # 使用精确模式对文本进行分词
    counts = {}

    for word in words:
        if len(word) == 1:  # 对分隔后的文本进行计数，单个词语不计算在内
            continue
        else:
            counts[word] = counts.get(word, 0) + 1

    print(counts)  # 键值对：{'数据': 55, '标签': 29, '可以': 24, '学习': 22, '模型': 22}

    items = list(counts.items())  # 将键值对转换成列表
    items.sort(key=lambda x: x[1], reverse=True)  # 根据词语出现的次数对列表进行倒序排列，list.sort()，x[0]=word,x[1]=count

    print(items)  # 列表：[('数据', 55), ('标签', 29), ('可以', 24), ('学习', 22), ('模型', 22)]

    # 输出前 20
    for i in range(20):
        word, count = items[i]  # 分别将 items[i].word 和 items[i].count 赋值给变量
        print("{0:<5}{1:>5}".format(word, count))
        # 数据 55
        # 标签 29
        # 可以 24


# 2、使用 DataFrame 存储词语及其出现次数
def show_txt_words_f2():
    # txt = open("C:\\Users\\Zero\\Desktop\\tools\\Python数据科学手册_机器学习.md", "r", encoding='utf-8').read()  # Windows
    txt = open("/Users/kinyuchung/Desktop/俯首为臣.txt", "r", encoding='gbk').read()  # MacOS
    words = jieba.lcut(txt)  # 使用精确模式对文本进行分词
    counts = pd.DataFrame(columns=['word', 'count']).set_index('word')

    # 对分隔后的文本进行计数
    for word in words:
        if len(word) == 1:  # 单个词语不计算在内
            continue
        else:
            try:
                counts.loc[word] = counts.loc[word] + 1  # 如果该 index 对应的行已存在，则 count 值加一
            except:
                new_word = pd.DataFrame(data=[{'word': word, 'count': 1}]).set_index('word')  # data=[{..}]，count 初始化为 1
                counts = counts.append(new_word, ignore_index=False).sort_values(by='count', ascending=False)  # 不存在，插入

    # print(counts)

    # 输出前 20
    print(counts.iloc[:20])


def show_words():
    word1 = jieba.lcut('冷咖啡离开了杯垫，我忍住的情绪在很后面。')
    word2 = jieba.lcut_for_search('冷咖啡离开了杯垫，我忍住的情绪在很后面。')  # 搜索引擎模式
    print(word1)
    print(word2)


if __name__ == '__main__':
    # show_words()
    # show_txt_words_f1()
    show_txt_words_f2()

