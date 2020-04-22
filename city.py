import pandas as pd


ls_cols = ['id', 'level', 'name', 'sname']
ls_city = ['name', 'sname', 'pid']
ls_plm = ['11', '12', '31', '50']  # 直辖市：北京、天津、上海、重庆(不需要市)
ls_sar = ['81', '82']  # 特别行政区：香港、澳门(不需要市)
ls_ssp = ['71']  # 特别省份：台湾(需要市)

# 1、读取文件，获取省份代码
df_region = pd.read_excel('files/region.xls', index_col=0).astype('str')[ls_cols]  # 设置第一列为索引，列类型设置为str，只取上述指定列
df_region['pid'] = df_region['id'].apply(str).apply(lambda x: x[:2])  # 获取id前两位作为省份代码，先将id列转成str，再进行切片，否则报错

# 2、分别获取省份和城市，并将两部分关联起来
df_province = df_region[(df_region['level'] == '1')]
df_city = df_region[(df_region['level'] == '2') & ~(df_region['pid'].isin(ls_plm + ls_sar))].append(df_province[(df_province['pid'].isin(ls_plm + ls_sar))])
df_result = pd.merge(df_province[ls_city], df_city[ls_city], on='pid', how='inner')
df_result.rename(columns={'name_x': 'pname', 'sname_x': 'psname', 'name_y': 'cname', 'sname_y': 'csname'}, inplace=True)  # 在原df上面修改

# 3、处理城市简称=省份+城市
# 判断A列取B列：df.loc()，如果pid满足条件就取psname，其余为NaN
df_result['rsname'] = df_result.loc[~(df_result['pid'].isin(ls_plm + ls_sar)), 'psname']  # 省份(直辖市和特别行政区不需要，这一列变成NaN)
df_result['rsname'] = df_result['rsname'].fillna('', inplace=True) + df_result['csname']  # 城市(先将前面为NaN的值用fillna填充掉)
# 判断同一列：df[col].apply()
df_result['rsname'] = df_result['pid'].apply(lambda x: '中国' if x in(ls_sar + ls_ssp) else '') + df_result['rsname']  # 港澳台前面加上中国

df_result.to_excel('files/city.xls')
