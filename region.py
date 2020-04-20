# Time : 2020/4/19 4:58 下午 

# Author : zhongjr

# File : region.py 

# Purpose: region.xls

import pandas as pd

columns = ['id', 'name', 'pid', 'sname', 'level', 'citycode', 'zipcode', 'mername', 'lng', 'lat', 'phonetic']
df_region = pd.DataFrame(columns=columns)
df = pd.DataFrame(columns=columns)
# df_region = df = pd.DataFrame(columns=columns)
# 不能这样写，这样会创建一个df_region到df的引用，df的值改变的话，df_region也会跟着改变，会导致后面数据重复
cnt = 0

with open('files/region.sql', mode='r', encoding='utf-8') as f:
    lines = f.readlines()  # 直接用readlines()分割换行
    for line in lines:
        if line.find('INSERT') == 0:
            data = line.replace('INSERT INTO `region` VALUES (', '').replace(');', '').replace("'", '').split(', ')
            df.loc[cnt] = data
            cnt = cnt + 1
            if cnt % 100 == 0:
                df_region = df_region.append(df, ignore_index=True)  # 分批提交
                df.drop(index=df.index, inplace=True)  # 清空df的行
                print(cnt)

df_region = df_region.append(df)
df_region.to_excel('files/region.xls')
print('write down:', cnt)
