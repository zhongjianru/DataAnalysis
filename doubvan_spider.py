import time
import re
import pandas as pd
import numpy as np

from selenium import webdriver


class douban_spider():
    def __init__(self):
        opt = webdriver.ChromeOptions()  # 创建chrome参数对象
        opt.add_argument('--headless')  # 无界面模式
        self.driver = webdriver.Chrome(options=opt)
        self.username = ''  # 账号
        self.password = ''  # 密码
        self.get_comments()

    def get_comments(self):
        # 打开豆瓣
        driver = self.driver
        driver.get("http://www.douban.com/")
        driver.switch_to.frame(driver.find_elements_by_tag_name("iframe")[0])
        # 点击"密码登录"
        bottom1 = driver.find_element_by_xpath('/html/body/div[1]/div[1]/ul[1]/li[2]')
        bottom1.click()
        # 输入账号密码
        input1 = driver.find_element_by_xpath('//*[@id="username"]')
        input1.clear()
        input1.send_keys(self.username)
        input2 = driver.find_element_by_xpath('//*[@id="password"]')
        input2.clear()
        input2.send_keys(self.password)
        # 登录
        bottom = driver.find_element_by_class_name('account-form-field-submit ')
        bottom.click()
        time.sleep(1)

        # sort=new_score按照热门程度排序，sort=time按照评论时间倒序，替换url中的start参数
        movieid = '26282488'
        url = 'https://movie.douban.com/subject/' + movieid + '/comments?start={}&limit=20&sort=new_score&status=P'

        compath = '//*[@id="comments"]/div[{}]/div[2]'
        path = {
            'cid': '//*[@id="comments"]/div[{}]',
            'type': compath + '/h3/span[2]/span[1]',
            'score':  compath + '/h3/span[2]/span[2]',
            'date':  compath + '/h3/span[2]/span[3]',
            'username': compath + '/h3/span[2]/a',
            'comment': compath + '/p/span',
            'votes': compath + '/h3/span[1]/span[1]',
            'userpage': compath + '/h3/span[2]/a',
            'location': '//*[@id="profile"]/div/div[2]/div[1]/div/a',
        }

        columns = ['cid', 'type', 'score', 'attitude', 'date', 'username', 'comment', 'votes', 'userpage', 'location']
        df_comments = pd.DataFrame(columns=columns)

        pattern = re.compile('(allstar)|(0 rating)')  # 将评分处理为1-5星
        page = 20  # 爬取页数
        pages = list(range(1, page+1))  # 评论页数列表
        step = 20  # 每页评论数，豆瓣默认为20个
        steps = list(range(1, step+1))  # 评论序号列表

        # 1、跟这里一样，一行一行循环获取，循环数=评论数，缺点：速度很慢，应该跟每次都获取driver有关，记得设置sleep，否则会被封号
        # 2、整页评论一起获取，获得多个list，再组成dataframe，循环数=列数*页数
        # 3、改进：先登录，获取useragent和cookie，直接用requests模块登录，再进行处理（driver可以相对url，requests要绝对url）

        for p in pages:
            print('正在获取第[', p, ']页评论')
            driver.get(url.format((p-1)*step))
            df_pagecmts = pd.DataFrame(columns=columns)  # 每获取一页评论，都清空之前的评论
            for i in steps:
                try:
                    cid = driver.find_element_by_xpath(path['cid'].format(str(i))).get_attribute('data-cid')
                    type = driver.find_element_by_xpath(path['type'].format(str(i))).text
                    score = driver.find_element_by_xpath(path['score'].format(str(i))).get_attribute('class')
                    score = re.sub(pattern=pattern, string=score, repl='')  # 将评分处理为1-5星
                    attitude = driver.find_element_by_xpath(path['score'].format(str(i))).get_attribute('title')  # 推荐程度
                    date = driver.find_element_by_xpath(path['date'].format(str(i))).get_attribute('title')
                    username = driver.find_element_by_xpath(path['username'].format(str(i))).text
                    comment = driver.find_element_by_xpath(path['comment'].format(str(i))).text
                    votes = driver.find_element_by_xpath(path['votes'].format(str(i))).text
                    userpage = driver.find_element_by_xpath(path['userpage'].format(str(i))).get_attribute('href')
                    location = ''
                except Exception as e:
                    continue  # 部分评论缺了一些要素，可以直接跳过

                try:
                    driver.get(userpage)
                    location = driver.find_element_by_xpath(path['location'].format(str(i))).text
                except Exception as e:
                    pass

                values = [cid, type, score, attitude, date, username, comment, votes, userpage, location]
                df_pagecmts.loc[i] = values
                # onecmt = dict(zip(columns, values))
                # onecmt = [comments, values]  # 不要总是想着写循环

                print('用户[', username, ']，居住地[', location, ']，评论为[', comment, ']。')
                driver.back()  # 返回上一个页面，一定要写，不写会报找不到element
                time.sleep(2)

            df_comments = df_comments.append(df_pagecmts, ignore_index=True)  # 追加本页评论
            df_comments.to_excel('comments.xls')  # 每页保存一次评论

        print('write down.')


if __name__ == '__main__':
    # pageNum = int(input("请输入想要爬取的页数： "))
    comments = douban_spider()
