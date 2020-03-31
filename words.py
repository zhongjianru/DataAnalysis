import jieba
import pandas as pd


def show_txt_words():
    txt = open("C:\\Users\\Zero\\Desktop\\tools\\Python数据科学手册_机器学习.md", "r", encoding='utf-8').read()
    words = jieba.lcut(txt)  # 使用精确模式对文本进行分词
    # counts = {}  # 通过键值对的形式存储词语及其出现的次数
    counts = pd.DataFrame(columns=['word', 'count']).set_index('word')

    # 对分隔后的文本进行计数
    for word in words:
        if len(word) == 1:  # 单个词语不计算在内
            continue
        else:
            # counts[word] = counts.get(word, 0) + 1  # 遍历所有词语，每出现一次其对应的值加 1
            # counts.loc[word] = 1
            counts.loc[word] = counts.loc[word] + 1  # KeyError

    print(counts)

    # items = list(counts.items())  # 将键值对转换成列表
    # items.sort(key=lambda x: x[1], reverse=True)  # 根据词语出现的次数进行从大到小排序

    # 输出前 15
    # for i in range(15):
    #     word, count = items[i]
    #     print("{0:<5}{1:>5}".format(word, count))


def show_words():
    word1 = jieba.lcut('冷咖啡离开了杯垫，我忍住的情绪在很后面。')
    word2 = jieba.lcut_for_search('冷咖啡离开了杯垫，我忍住的情绪在很后面。')  # 搜索引擎模式
    print(word1)
    print(word2)


if __name__ == '__main__':
    # show_words()
    show_txt_words()

