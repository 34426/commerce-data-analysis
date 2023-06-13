#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

plt.rcParams['font.sans-serif'] = ['SimHei']  #设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False #正常显示负号

pd.set_option('display.float_format',lambda x : '%.2f' % x)#pandas禁用科学计数法

#忽略警告
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# 数据清洗
# 1.查看数据基本信息
# 2.字段介绍
# 3.创建新列-日期、月份、小时、周几
# 4.查看数据缺失、重复情况
# 5.查看数据是否有异常
# 6.保存清洗后的数据


# In[ ]:


# 1.查看数据基本信息


# In[2]:


data = pd.read_csv( r"C:\Users\fwl\Desktop\电商用户消费行为分析数据集处理\海量数据预处理实战\电子产品销售数据.csv",index_col=0,dtype={'category_id':'int64','user_id':'int64'},encoding='utf-8',sep=',')


# In[3]:


data.head()


# In[4]:


# 数据框大小
data.shape


# In[13]:


# 数据基本信息
data.info()


# In[ ]:


# 2.字段介绍
# Unnamed: 行号
# event_time：下单时间
# order_id：订单编号
# product_id:产品标号
# category_id :类别编号
# category_code :类别
# brand :品牌
# price :价格
# user_id :用户编号
# age :年龄
# sex :性别
# local:省份


# In[ ]:


# 3.创建新列-日期、月份、小时、周几


# In[ ]:


# 创建日期列


# In[6]:


data['date'] = data.event_time.apply(lambda x: x.split(' ')[0])


# In[7]:


#转换为日期格式
data['date'] = pd.to_datetime(data['date'])


# In[ ]:


# 创建月份列


# In[8]:


data['month'] = data.date.dt.month


# In[ ]:


# 创建"小时"列


# In[9]:


data['hour'] = data.event_time.apply(lambda x: x.split(' ')[1].split(':')[0])


# In[ ]:


# 创建周几列---周日为0,周一为1


# In[10]:


data['weekday'] = data.date.apply(lambda x:x.strftime("%w"))


# In[ ]:


# 删除event_time列


# In[11]:


del data['event_time']


# In[12]:


data.head()


# In[ ]:


# 4.查看数据缺失、重复情况


# In[ ]:


# 4.1查看数据缺失并删除缺失值的数据
# 4.2存在重复值,但是换个角度去想,这些重复值就是同笔订单下了多个数量的订单,所以不删除重复值,进而增加一列购买数量的列和总价的列


# In[14]:


data.shape


# In[15]:


data.info()


# In[ ]:


# 缺失数据有category_code-产品类别和brand-品牌这两列，对于category_code用"R"来代替缺失值而不是选择删除缺失值的数据

# brand这一列数据缺失比较少,直接删除缺失值


# In[16]:


data['category_code'] = data['category_code'].fillna("R")


# In[17]:


#删除brand这一列有缺失值的数据
data = data[data.brand.notnull()]


# In[18]:


data.info()


# In[ ]:


# 4.2存在重复值,但是换个角度去想,这些重复值就是同笔订单下了多个数量的订单,所以不删除重复值,进而增加一列购买数量的列和总价的列


# In[19]:


data.duplicated().sum()


# In[20]:


data.duplicated()


# In[21]:


#添加新的列:购买数量
data = data.value_counts().reset_index().rename(columns={0:'buy_cnt'})
#由于python版本问题,旧的版本没有上面的功能,所以要写以下3行代码
# df = data.groupby(['order_id','product_id']).agg(buy_cnt=('user_id','count'))
# data = pd.merge(data,df,on=['order_id','product_id'],how='inner')
# data = data.drop_duplicates().reset_index(drop=True)


# In[22]:


#添加新的列:购买总金额
data['amount'] = data['price'] * data['buy_cnt']


# In[ ]:


# 5.查看数据是否有异常


# In[ ]:


# 5.1把几个id的格式转化为object格式
# 5.2把hour和weekday转化为int
# 5.3查看价格和年龄是否存在异常值
# 5.4检查其他字段是否有异常值
# 5.4.1发现date日期有异常值,显示为1970-01-01,把这些异常值删除


# In[ ]:


# 5.1把几个id的格式转化为object格式


# In[23]:


data.order_id = data.order_id.astype('object')
data.product_id = data.product_id.astype('object')
data.category_id = data.category_id.astype('object')
data.user_id = data.user_id.astype('object')


# In[ ]:


# 5.2把hour和weekday转化为int


# In[24]:


data['hour'] = data.loc[:,'hour'].astype('int')
data['weekday'] = data.loc[:,'weekday'].astype('int')


# In[25]:


data.info()


# In[ ]:


# 5.3查看价格和年龄是否存在异常值


# In[26]:


data.describe(percentiles=[0.01,0.25,0.75,0.99]).T


# In[ ]:


# 以上7个字段均没有异常值
# price和amount最小值为0，这类商品应该就是免费类的商品，所以也不属于异常值。
# 应该进一步分析，购买了0元商品的用户，后续是否还有购买了其他的商品。


# In[ ]:


# 5.4检查其他字段是否有异常值


# In[27]:


data.describe(include='all').T


# In[ ]:


# 5.4.1发现date日期有异常值,显示为1970-01-01,把这些异常值删除


# In[28]:


data = data[data.date>'1970-01-01']


# In[29]:


data.date.min()


# In[ ]:


# 6.保存清洗后的数据


# In[30]:


data.head()


# In[31]:


data.shape


# In[33]:


# 重置索引
data.reset_index(drop=True,inplace=True)


# In[32]:


#保存清洗后的数据
data.to_csv('C:\Users\fwl\Desktop\电商用户消费行为分析数据集处理\海量数据预处理实战\data_clean.csv',index=False)


# In[ ]:




