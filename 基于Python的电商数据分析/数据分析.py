#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[3]:


data = pd.read_csv( r"C:\Users\fwl\Desktop\电商用户消费行为分析数据集处理\海量数据预处理实战\data_clean.csv",encoding='utf-8',sep=',')


# In[4]:


data.head()


# In[5]:


# 1.数据分析
# 总的指标
# 总GMV:约1.15亿元
# 每月的GMV  
# 客单价:1240元


# In[6]:


# 6.1.1总GMV:约1.15亿元


# In[7]:


round(data['amount'].sum(),0)


# In[8]:


# 6.1.2每月的GMV  


# In[9]:


# GMV8月之前都基本是处于上升状态,在7月8月的上升更是非常大,8月达到峰值,然后就开始下降了


# In[10]:


GMV_month = data.groupby('month').agg(GMV=('amount','sum'))
GMV_month


# In[11]:


plt.plot(GMV_month.index,GMV_month['GMV'])
plt.show()


# In[12]:


# 6.1.3客单价:1240元


# In[13]:


#按客户数量
round(data['amount'].sum() / data['user_id'].nunique(),0)


# In[14]:


#按订单数量
round(data['amount'].sum() / data['order_id'].nunique(),0)


# In[15]:


# 产品分析
# 1.销售第一的产品类别的销售前五品牌分析
# （1）销量\销售额前十产品
# （2）销量少于10的产品
# （3）销量前十的产品类别 category_code
# （4）对于手机,销量前五的品牌-brand
# 2.销售额前十品牌产品消费总金额


# In[16]:


# 销售额前十产品


# In[17]:


amount = data.groupby('product_id').agg(销售总额=('amount','sum')).reset_index().sort_values('销售总额',ascending=False).reset_index(drop=True)
amount.head(10)


# In[18]:


# 销量前十产品
# 对于热销产品,应该时刻关注他们的库存量,避免发生缺货情况


# In[19]:


cnt = data.groupby('product_id').agg(销售总量=('buy_cnt','sum')).reset_index().sort_values('销售总量',ascending=False).reset_index(drop=True)
cnt.head(10)


# In[20]:


#销量少于10的产品
# 有12069个产品销量少于10,对于这批产品可以考虑促销活动对其进行清仓处理


# In[21]:


cnt.describe(percentiles=(0.01,0.1,0.25,0.75,0.9,0.99))
cnt[cnt.销售总量<10]


# In[22]:


# 销量前十的产品类别 category_code
# 最受欢迎的产品类别是smartphone-即手机,是第二名手提电脑的4倍
# 需要去除类别为R的,因为是缺失数据


# In[23]:


cnt_category = data[data.category_code != 'R'].groupby('category_code').agg(销量=('buy_cnt','sum')).reset_index().sort_values('销量',ascending=False).reset_index(drop=True)
cnt_category.head(10)


# In[24]:


# 对于手机,销量前五的品牌-brand
# A


# In[25]:


brand_5 = data[data.category_code=='electronics.smartphone'].groupby('brand').agg(销量=('buy_cnt','sum')).reset_index().sort_values('销量',ascending=False)
brand_5.reset_index(drop=True,inplace=True)
brand_5.head(5)


# In[26]:


brand_5['销量'].sum()


# In[27]:


plt.pie(data=brand_5.head(5)
        ,x='销量'
        ,labels='brand'
        ,autopct='%.1f%%'
        ,textprops={'fontsize':12, 'color':'k'} # 设置文本标签的属性值
        ,radius=2
        )
plt.show()


# In[28]:


# 2.各品牌产品消费总金额


# In[29]:


dbs=data.groupby('brand').sum().sort_values('price',ascending=False).head(10)
# dbs = data.groupby('brand').agg(销售总量=('buy_cnt','sum')).reset_index().sort_values('销售总量',ascending=False).reset_index(drop=True)
# dbs.head(10)
plt.figure(figsize=(10,6),dpi=100)
x1=dbs.index.to_list()
y1=dbs['price']
# plt.figure(figsize=(30,24))
plt.bar(x1,y1,facecolor='pink',label='各品牌销售数据')
plt.xlabel('品牌',fontsize=18)
plt.ylabel('销售额(千万)',fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=18)
for x1, y1 in zip(x1, y1):
    plt.text(x1, y1 , '%.2f' % y1, ha='center', va='bottom',fontsize=10)
plt.show()


# In[30]:


# 结论：
# （1）得出了销量前10和销售额前10的产品，要时刻关注他们的库存量，避免发生缺货现象。
# （2）对于产品销量少于10的产品，可以考虑促销活动对其进行清仓处理。
# （3）最受欢迎的产品类别是smartphone-即手机,是第二名手提电脑的4倍。
# （4）手机销量前五分别为三星、苹果、小米、华为、OPPO，其中三星占了一半以上的份额,苹果约占据了四分之一。
# （5）销售前10名品牌中，三星和苹果表现尤为强势，远超其他品牌。


# In[31]:


# 1.各省销量、销售额情况
# 2.按日期分析销售额与销量的总体走势
# 3.按月份对销售额和销量进行分析
# （1）新客老客的销售额与销量对比
# （2）新老客户销售额对比图
# （3）新老客户销量对比图
# 4.每月的新客、老客复购人数分析(老客定义:首次购买次月即成为老客)


# In[32]:


# 1.各省销量、销售额情况


# In[33]:


local_situation = data.groupby('local').agg(销量=('buy_cnt','sum'),销售额=('amount','sum')).sort_values('销售额')
local_situation['销售额'] = local_situation['销售额'].astype('int')


# In[34]:


plt.figure(figsize=(16,8))
plt.barh(local_situation.index,local_situation['销售额'])
for i,j in enumerate(local_situation['销售额']):
    plt.text(j+200,local_situation['销售额'].index[i],j)
plt.title('销售额')
plt.show()


# In[35]:


plt.figure(figsize=(16,8))
plt.barh(local_situation.index,local_situation['销量'])
for i,j in enumerate(local_situation['销量']):
    plt.text(j+200,local_situation['销量'].index[i],j)
plt.title('销量')
plt.show()


# In[36]:


# 2.按日期分析销售额与销量的总体走势


# In[37]:


# 数据显示,618和双十一当天的销量和销售额并没有很高,反而处于低位(理论上不可能存在这种情况的-数据是否不够真实!!!)
# 假设数据真实，那么可以证实618和双十一的活动效果是非常差的，进行复盘
# 结合后面的分析，公司处于起步阶段，所以618与双十一的活动很难和其他已经开了好几年的公司拼。


# In[38]:


data_date = data.groupby('date').agg(销售额=('amount','sum'),销量=('buy_cnt','sum'))
plt.figure(figsize=(20,5))
plt.plot(data_date.index,data_date['销售额'])
plt.xticks([]) #隐藏坐标轴
plt.xticks([data_date.index.min(),'2020-06-18','2020-11-11'])
plt.title('销售额')
plt.show()


# In[39]:


plt.figure(figsize=(20,5))
plt.plot(data_date.index,data_date['销量'])
plt.xticks([]) #隐藏坐标轴
plt.xticks([data_date.index.min(),'2020-06-18','2020-11-11'])
plt.title('销量')
plt.show()


# In[ ]:


# 3.按月份对销售额和销量进行分析
# （1）新客老客的销售额与销量对比
# （2）新老客户销售额对比图
# （3）新老客户销量对比图


# In[ ]:


# 销售额1-8月呈上升趋势,但是8月份之后就开始下降
# 销量基本与销售额呈一致的趋势


# In[71]:


data_month = data.groupby('month').agg(销售额=('amount','sum'),销量=('buy_cnt','sum'))


# In[72]:


plt.figure(figsize=(16,5))
plt.plot(data_month.index,data_month['销售额'])
plt.title('销售额')
plt.xticks(data_month.index)
plt.show()


# In[73]:


plt.figure(figsize=(16,5))
plt.plot(data_month.index,data_month['销量'])
plt.title('销量')
plt.xticks(data_month.index)
plt.show()


# In[ ]:


# 新客老客的销售额与销量对比


# In[75]:


#划分每个用户的首次购买月份(用来确认用户在几月份是属于新客户)
data_user = data.groupby('user_id').agg(首次购买月份=('month','min')).reset_index()
user_all = pd.merge(data,data_user,on='user_id')
user_all['新老客户'] = np.where(user_all['month']==user_all['首次购买月份'],'新客户','老客户')
# user_all.head()

#每月新客的销售额和销量
user_all_new = user_all[user_all['新老客户']=='新客户'].groupby('month').agg(销售额=('amount','sum'))
user_all_new['销量'] = user_all[user_all['新老客户']=='新客户'].groupby('month').agg(销量=('buy_cnt','sum'))
# user_all_new

user_all_old = user_all[user_all['新老客户']=='老客户'].groupby('month').agg(销售额=('amount','sum'))
user_all_old['销量'] = user_all[user_all['新老客户']=='老客户'].groupby('month').agg(销量=('buy_cnt','sum'))
# user_all_old


# In[76]:


#每月新客的销售额和销量
user_all_new = user_all[user_all['新老客户']=='新客户'].groupby('month').agg(销售额=('amount','sum'))
user_all_new['销量'] = user_all[user_all['新老客户']=='新客户'].groupby('month').agg(销量=('buy_cnt','sum'))
user_all_new


# In[77]:


user_all_old = user_all[user_all['新老客户']=='老客户'].groupby('month').agg(销售额=('amount','sum'))
user_all_old['销量'] = user_all[user_all['新老客户']=='老客户'].groupby('month').agg(销量=('buy_cnt','sum'))
user_all_old


# In[ ]:


# 新老客户销售额对比图


# In[ ]:


# 从数据中发现，基本每个月都是新客的贡献度都大于老客，证明公司处于起步阶段，老客户的黏性还不够高


# In[78]:


plt.figure(figsize=(16,8))
line1, = plt.plot(user_all_new.index,user_all_new['销售额'],c='r')
line2, = plt.plot(user_all_old.index,user_all_old['销售额'],c='b')
plt.legend([line1,line2],['新客','老客'])
plt.xticks(user_all_new.index)
plt.title('新老客户销售额对比')
plt.show()


# In[ ]:


# 新老客户销量对比图


# In[94]:


plt.figure(figsize=(16,8))
line1, = plt.plot(user_all_new.index,user_all_new['销量'],c='r')
line2, = plt.plot(user_all_old.index,user_all_old['销量'],c='b')
plt.legend([line1,line2],['新客','老客'])
plt.xticks(user_all_new.index)
plt.title('新老客户销量对比')
plt.show()


# In[ ]:


# 4.每月的新客、老客复购人数分析(老客定义:首次购买次月即成为老客)


# In[96]:


data_buy = data.groupby(['user_id','month','date']).agg(是否购买=('user_id','nunique')).reset_index()
data_buy_month = data_buy.groupby(['user_id','month']).agg(每月购买次数=('是否购买','sum')).reset_index()


# In[101]:


data_repurchase = data_buy_month[data_buy_month['每月购买次数']>=2].groupby('month').agg(每月复购人数=('user_id','nunique'))
data_repurchase['每月购买人数'] = data.groupby('month').agg(每月购买人数=('user_id','nunique'))
data_repurchase['复购率'] = data_repurchase['每月复购人数'] / data_repurchase['每月购买人数']


# In[97]:


#划分每个用户的首次购买月份(用来确认用户在几月份是属于新客户)
data_user = data.groupby('user_id').agg(首次购买月份=('month','min')).reset_index()

#data_buy_month 是上面求得的:每个用户每月的购买次数
user = pd.merge(data_buy_month,data_user)
user['新老客户'] = np.where(user['month']==user['首次购买月份'],'新客户','老客户')
#新客户购买人数
user_buy = user[user['新老客户']=='新客户'].groupby('month').agg(新客购买人数=('user_id','nunique'))
#老客户购买人数
user_buy['老客购买人数'] = user[user['新老客户']=='老客户'].groupby('month').agg(老客购买人数=('user_id','nunique'))
user_buy = user_buy.fillna(0)
user_buy


# In[ ]:


#划分每个用户的首次购买月份(用来确认用户在几月份是属于新客户)
data_user = data.groupby('user_id').agg(首次购买月份=('month','min')).reset_index()


# In[98]:


#data_buy_month 是上面求得的:每个用户每月的购买次数
user = pd.merge(data_buy_month,data_user)
user['新老客户'] = np.where(user['month']==user['首次购买月份'],'新客户','老客户')
#筛选出每月购买次数大于等于2的数据
user_2 = user[user['每月购买次数'] >= 2]


# In[99]:


#每月新客户的复购人数
user_repurchase = user_2[user_2['新老客户']=='新客户'].groupby('month').agg(新客户复购人数=('user_id','nunique'))
# user_repurchase
#每月老客户的复购人数
#1月只有新客户 所以老客户复购人数为0
user_repurchase['老客户复购人数'] = user_2[user_2['新老客户']=='老客户'].groupby('month').agg(老客户复购人数=('user_id','nunique'))
user_repurchase = user_repurchase.fillna(0)
user_repurchase['老客户复购人数']  = user_repurchase['老客户复购人数'].astype('int')


# In[104]:


plt.subplots(figsize=(16,5))
line1, = plt.plot(user_repurchase.index,user_repurchase['新客户复购率'],c='r')
line2, = plt.plot(user_repurchase.index,user_repurchase['老客户复购率'])
plt.title('每月新老客户的复购率')
plt.xticks(user_repurchase.index)
plt.legend([line1,line2],['新客户复购率','老客户复购率'],loc=4)
plt.show()


# In[ ]:


# 第一次复购人数增长较多是在5月份，贡献最多的是新客（新客是老客的2倍多）；
# 第二次复购人数增长较多是在7月份，8月份也上涨了，贡献最多的依然是新客；
# 然后9月份开始下降，新客复购人数下降了63%，老客仅下降了13%，贡献率第一次被老客反超新客；
# 复购率1-3月在上升,然后4月份下降到比1月份还低，进一步分析，4月份进行了拉新活动，较多新客当月只购买了一次（有可能是因为拉新活动启动比较晚，新客的复购周期还没到；也有可能这品新客在4月的购买满意度较低，应该着重分析这批客户是否有投诉），所以造成复购率下降；
# 复购率5月份开始上升,直至11月份下降，11月份新客户的复购率大幅下降，而从9月开始，新客与老客的复购人数都在减少，加上新客数的骤降就导致复购率下降了。老客的复购率都在下降，那么很可能是老客对我们店铺的满意度不够，应该做好产品的优化与服务，维系好老客，还有加强拉新。


# In[ ]:


# 销售情况分析：
# 广东、上海、北京的销售额、销量以及客户数量都是最高的。
# 销售额与销量1-8月呈上升趋势，8月达到峰值，5、7、8月上升较大，这三个月的新客户都有增长，幅度也不少，证明拉新活动效果不错；
# 9月销售额与销量和新客都开始下降，其中新客的下单量减少了四分之三，证明9月份的拉新活动效果非常差。同时发现618和双十一当天的销售情况非常差。


# In[ ]:


# 应该继续进行7月8月的拉新活动的投入，加大在广东的投入。
# 清仓处理销量很低的产品，也可以将价值较低的产品作为高价值产品的赠送物品；
# 对于销量高的产品，要保证库存率在合理水平（太高会占用较多的流动资金和增加仓库费用，太低会容易导致缺货）。
# 根据店铺目前的情况来看，销量最高的是手机，故应以手机为主推产品（用以引流）。


# In[ ]:




