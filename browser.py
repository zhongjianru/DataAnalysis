# Time : 2020/4/12 11:35 下午 

# Author : zhongjr

# File : browser.py 

# Purpose: selenium模拟登陆

import time
from selenium import webdriver

# 'chromedriver' executable needs to be in PATH
# https://blog.csdn.net/weixin_41990913/article/details/90936149
# https://www.jianshu.com/p/4889dd130f76 (MacOS)

# Selenium及Headless Chrome抓取动态HTML页面
# https://www.cnblogs.com/linxiyue/p/10215912.html

# Selenium模拟登录并获取session
# https://blog.csdn.net/wufeil7/article/details/85224613

# 保存DataFrame之前要做的工作
# https://blog.csdn.net/weixin_43952650/article/details/89296710


def simulate_f1():

    browser = webdriver.Chrome()
    browser.get('https://www.baidu.com')

    # 执行完脚本后浏览器会自动退出


# headless模式
def simulate_f2():

    # 加启动配置
    option = webdriver.ChromeOptions()
    option.add_argument('--no-sandbox')
    option.add_argument('--headless')
    option.add_argument('--disable-gpu')

    # 打开chrome浏览器
    browser = webdriver.Chrome(options=option)
    browser.get("https://www.baidu.com")

    # 获取的源码都是些js与css语句，dom并未生成，需要模拟浏览器滚动来生成dom：
    for i in range(1, 11):
        browser.execute_script(
            "window.scrollTo(0, document.body.scrollHeight/10*%s);" % i
        )
        time.sleep(0.5)

    data = browser.page_source  # 返回的源代码
    print(data)


if __name__ == '__main__':
    # simulate_f1()
    simulate_f2()