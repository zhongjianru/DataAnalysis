# Time : 2020/4/12 11:34 下午 

# Author : zhongjr

# File : taobao.py 

# Purpose: 部分网页可以保存到本地再进行解析，但是相对路径url不能直接访问

import os
import time

from selenium import webdriver
from pyquery import PyQuery as pq


def get_items():

    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--disable-gpu')
    browser = webdriver.Chrome(options=chrome_options)
    # browser.implicitly_wait(10)
    url = 'http://www.taobao.com'   # 保存下来的页面乱码
    browser.get(url)

    # 获取的源码都是些js与css语句，dom并未生成，需要模拟浏览器滚动来生成dom
    for i in range(1, 11):
        browser.execute_script(
            "window.scrollTo(0, document.body.scrollHeight/10*%s);" % i
        )
        time.sleep(0.5)
    data = browser.page_source.encode('utf-8')

    # 网页为了让img延迟加载，img的地址是放在data-img属性上的，等到浏览器滑动至图片时才修改src属性，可以使用pyquery修改
    doc = pq(data)
    for img in doc('img'):
        img = pq(img)
        if img.attr['data-img']:
            img.attr.src = img.attr['data-img']
    data = doc.html(method='html').replace('src="//', 'src="http://')
    base_dir = os.path.dirname(__file__)
    file_dir = os.path.join(base_dir, 'taobao.html')
    f = open(file_dir, 'wb')  # 不用自己拼接路径，可以使用函数拼接
    f.write(data.encode('utf-8'))
    f.close()
    print('page saved at:', file_dir)


if __name__ == '__main__':
    get_items()
