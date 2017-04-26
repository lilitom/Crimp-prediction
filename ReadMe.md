# 携程出行产品未来14个月销量预测
## 数据的分析
> 
##### In1
```
#coding =utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime 
from sklearn.ensemble import BaggingRegressor
import sys
#读取数据
evaluation=pd.read_csv('prediction_lilei_20170320.txt')
product_info=pd.read_csv('product_info.txt')#酒店的参数
product_quantity=pd.read_csv('product_quantity.txt')#酒店的一些销售信息等
```
##### In2
```
#数据的信息
#官方给出-1代表1，这里把-1替换为NAN
product_info=product_info.replace(-1,np.nan)
product_quantity=product_quantity.replace(-1,np.nan)
product_info.info()#看看里面的有没有缺失值
#看到
'''
railway          177 non-null float64
airport          178 non-null float64
citycenter       264 non-null float64
railway2         122 non-null float64
airport2         122 non-null float64
citycenter2      122 non-null float64
'''
# 大量缺失数据，可以Drop掉
```
##### Out2

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 4000 entries, 0 to 3999
Data columns (total 22 columns):
product_id       4000 non-null int64
district_id1     4000 non-null int64
district_id2     3995 non-null float64
district_id3     4000 non-null int64
district_id4     3250 non-null float64
lat              3896 non-null float64
lon              3896 non-null float64
railway          177 non-null float64
airport          178 non-null float64
citycenter       264 non-null float64
railway2         122 non-null float64
airport2         122 non-null float64
citycenter2      122 non-null float64
eval             4000 non-null float64
eval2            3995 non-null float64
eval3            3995 non-null float64
eval4            3992 non-null float64
voters           3992 non-null float64
startdate        4000 non-null object
upgradedate      4000 non-null object
cooperatedate    4000 non-null object
maxstock         3995 non-null float64
dtypes: float64(16), int64(3), object(3)
memory usage: 687.6+ KB
Out[36]:
'\nrailway          177 non-null float64\nairport          178 non-null float64\ncitycenter       264 non-null float64\nrailway2         122 non-null float64\nairport2         122 non-null float64\ncitycenter2      122 non-null float64\n'
```
##### In3

```
#再来看看具体的信息的描述
product_info.describe()
```

##### Out3

```
	product_id	district_id1	district_id2	district_id3	district_id4	lat	lon	railway	airport	citycenter	railway2	airport2	citycenter2	eval	eval2	eval3	eval4	voters	maxstock
count	4000.000000	4000.000000	3.995000e+03	4.000000e+03	3.250000e+03	3896.000000	3896.000000	177.000000	178.000000	264.000000	122.000000	122.000000	122.000000	4000.000000	3995.000000	3995.000000	3992.000000	3992.000000	3995.000000
mean	2000.500000	10293.364500	3.459705e+04	1.588152e+05	4.723114e+05	12.691926	45.819982	12.802542	32.309551	8.851894	3.174590	3.274590	3.374590	3.813250	4.635795	2.915019	3.455235	1580.157816	168.580976
std	1154.844867	905.997621	1.172602e+05	4.487188e+05	5.624375e+05	9.645501	31.898967	20.263625	19.032698	14.958871	3.094238	3.094238	3.094238	1.015939	1.407388	1.271525	0.366572	2041.405022	176.454626
min	1.000000	10201.000000	2.040000e+04	3.100300e+04	4.201600e+04	0.000000	0.000000	0.300000	4.200000	0.400000	0.500000	0.600000	0.700000	1.000000	2.000000	0.000000	0.000000	2.000000	0.000000
25%	1000.750000	10201.000000	2.131800e+04	3.213600e+04	6.741800e+04	4.623181	22.650924	3.100000	20.200000	1.800000	1.400000	1.500000	1.600000	3.000000	4.000000	2.000000	3.400000	442.750000	73.000000
50%	2000.500000	10201.000000	2.203200e+04	3.532900e+04	1.387360e+05	11.742883	42.451267	6.100000	30.200000	4.300000	2.400000	2.500000	2.600000	3.000000	4.000000	2.500000	3.500000	929.500000	127.000000
75%	3000.250000	10201.000000	2.274600e+04	7.766200e+04	1.187264e+06	21.039767	76.989312	15.100000	40.200000	10.300000	3.400000	3.500000	3.600000	5.000000	5.000000	4.000000	3.600000	1869.250000	200.000000
max	4000.000000	24846.000000	1.307640e+06	4.154917e+06	1.567800e+06	42.068964	110.096590	130.100000	160.200000	146.300000	25.400000	25.500000	25.600000	6.000000	8.000000	6.500000	4.000000	25778.000000	3329.000000
```

##### In4

```
#大概看一下数据的样子
product_info.head()
```

##### Out4

```
	product_id	district_id1	district_id2	district_id3	district_id4	lat	lon	railway	airport	citycenter	...	citycenter2	eval	eval2	eval3	eval4	voters	startdate	upgradedate	cooperatedate	maxstock
0	1	10201	20502.0	31003	45760.0	3.994928	11.634630	NaN	NaN	NaN	...	NaN	3.0	5.0	2.0	3.1	1034.0	2005-11-01	2015-01-01	2013-07-02	75.0
1	2	10201	20502.0	31003	45760.0	3.995148	11.636258	NaN	NaN	2.3	...	NaN	3.0	4.0	2.0	3.4	1707.0	2005-02-28	2011-01-01	2014-12-16	172.0
2	3	10201	20502.0	31003	45760.0	3.994291	11.631246	NaN	NaN	10.3	...	NaN	3.0	4.0	2.5	3.6	1739.0	2007-03-01	2014-01-01	2014-07-02	188.0
3	4	10201	20502.0	31003	55952.0	3.997783	11.641561	NaN	NaN	9.3	...	NaN	3.0	4.0	2.5	3.5	1065.0	2006-07-01	1753-01-01	2014-12-19	116.0
4	5	10201	20502.0	31003	55952.0	3.999904	11.641149	12.1	25.2	13.3	...	3.6	3.0	5.0	2.0	3.4	2209.0	2007-01-01	2012-03-01	2007-11-07	95.0
5 rows × 22 columns
```

##### In5

```
##可以看到缺失的数据是很多的，我们先对一些不是缺失值很多的进行处理
product_info.loc[product_info['lat'].isnull(),'lat']=product_info['lat'].mean()
product_info.loc[product_info['lon'].isnull(),'lon']=product_info['lon'].mean()
product_info.loc[product_info['eval2'].isnull(),'eval2']=product_info['eval2'].mean()
product_info.loc[product_info['eval3'].isnull(),'eval3']=product_info['eval3'].mean()
product_info.loc[product_info['eval4'].isnull(),'eval4']=product_info['eval4'].mean()
product_info.loc[product_info['eval4'].isnull(),'eval4']=product_info['eval4'].mean()
product_info.loc[product_info['voters'].isnull(),'voters']=product_info['voters'].mean()
product_info.loc[product_info['maxstock'].isnull(),'maxstock']=product_info['maxstock'].mean()
product_info.loc[product_info['district_id2'].isnull(),'district_id2']=product_info['district_id2'].mean()
product_info.loc[product_info['district_id4'].isnull(),'district_id4']=product_info['district_id4'].mean()
```


##### In6

```
#再来查看
product_info.info()
#--->good!!!!
```

##### Out6

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 4000 entries, 0 to 3999
Data columns (total 22 columns):
product_id       4000 non-null int64
district_id1     4000 non-null int64
district_id2     4000 non-null float64
district_id3     4000 non-null int64
district_id4     4000 non-null float64
lat              4000 non-null float64
lon              4000 non-null float64
railway          177 non-null float64
airport          178 non-null float64
citycenter       264 non-null float64
railway2         122 non-null float64
airport2         122 non-null float64
citycenter2      122 non-null float64
eval             4000 non-null float64
eval2            4000 non-null float64
eval3            4000 non-null float64
eval4            4000 non-null float64
voters           4000 non-null float64
startdate        4000 non-null object
upgradedate      4000 non-null object
cooperatedate    4000 non-null object
maxstock         4000 non-null float64
dtypes: float64(16), int64(3), object(3)
memory usage: 687.6+ KB
```

##### In7

```
#对每个酒店每个月的销售量做一个统计
product_quantity['product_month']=product_quantity['product_date'].apply(lambda x: x[:7])
train_month=product_quantity.groupby(['product_id','product_month']).sum()['ciiquantity']
train_month=pd.DataFrame(train_month)
train_month=train_month.reset_index()
train_month.head(30)
```

##### Out7

```
	product_id	product_month	ciiquantity
0	1	2014-01	29
1	1	2014-02	111
2	1	2014-03	13
3	1	2014-04	71
4	1	2014-05	74
5	1	2014-06	30
6	1	2014-07	55
7	1	2014-08	159
8	1	2014-09	35
9	1	2014-10	134
10	1	2014-11	57
11	1	2014-12	51
12	1	2015-01	73
13	1	2015-02	39
14	1	2015-03	102
15	1	2015-04	283
16	1	2015-05	136
17	1	2015-06	52
18	1	2015-07	85
19	1	2015-08	48
20	1	2015-09	37
21	1	2015-10	102
22	1	2015-11	85
23	2	2014-01	46
24	2	2014-02	59
25	2	2014-03	104
26	2	2014-04	144
27	2	2014-05	167
28	2	2014-06	117
29	2	2014-07	194
```
## 特征的选取
##### In8

```
#保留有用的信息
product_info_use=product_info.drop(['railway', 'airport', 'citycenter', 'railway2', 'airport2','citycenter2',  'startdate', \
                                    'upgradedate', 'cooperatedate'],axis=1)
product_info_use.head(10)
```

##### Out8

```
	product_id	district_id1	district_id2	district_id3	district_id4	lat	lon	eval	eval2	eval3	eval4	voters	maxstock
0	1	10201	20502.0	31003	45760.0	3.994928	11.634630	3.0	5.0	2.0	3.1	1034.0	75.0
1	2	10201	20502.0	31003	45760.0	3.995148	11.636258	3.0	4.0	2.0	3.4	1707.0	172.0
2	3	10201	20502.0	31003	45760.0	3.994291	11.631246	3.0	4.0	2.5	3.6	1739.0	188.0
3	4	10201	20502.0	31003	55952.0	3.997783	11.641561	3.0	4.0	2.5	3.5	1065.0	116.0
4	5	10201	20502.0	31003	55952.0	3.999904	11.641149	3.0	5.0	2.0	3.4	2209.0	95.0
5	6	10201	20502.0	31003	55952.0	3.999534	11.642633	3.0	7.0	2.0	3.5	1788.0	150.0
6	7	10201	20502.0	31003	55952.0	3.998353	11.641247	3.0	7.0	2.0	3.6	1472.0	130.0
7	8	10201	20502.0	31003	55952.0	4.003753	11.641792	3.0	6.0	2.0	3.3	1796.0	88.0
8	9	10201	20502.0	31003	55952.0	4.000440	11.636935	3.0	7.0	2.0	3.4	1304.0	128.0
9	10	10201	20502.0	31003	55952.0	4.000185	11.642194	3.0	4.0	2.0	3.4	2356.0	210.0
```

##### In9


```
#归一化
def one_hot(table,name):
    dummies = pd.get_dummies(table[name], prefix=name, drop_first=False)
    table = pd.concat([table, dummies], axis=1)
    table = table.drop(name, axis=1)
    return table
def Pretreatment(table):
    table['year']=table['product_month'].apply(lambda x:(float(x[0:4])-2015.5)/4)
    table['month']=table['product_month'].apply(lambda x:x[5:7])
    table=one_hot(table,'month')
    table=table.drop('product_month', axis=1)    
    return table
train_month=Pretreatment(train_month)
train_month.head(50)
```

##### Out9

```
	product_id	ciiquantity	year	month_01	month_02	month_03	month_04	month_05	month_06	month_07	month_08	month_09	month_10	month_11	month_12
0	1	29	-0.375	1	0	0	0	0	0	0	0	0	0	0	0
1	1	111	-0.375	0	1	0	0	0	0	0	0	0	0	0	0
2	1	13	-0.375	0	0	1	0	0	0	0	0	0	0	0	0
3	1	71	-0.375	0	0	0	1	0	0	0	0	0	0	0	0
4	1	74	-0.375	0	0	0	0	1	0	0	0	0	0	0	0
5	1	30	-0.375	0	0	0	0	0	1	0	0	0	0	0	0
6	1	55	-0.375	0	0	0	0	0	0	1	0	0	0	0	0
7	1	159	-0.375	0	0	0	0	0	0	0	1	0	0	0	0
8	1	35	-0.375	0	0	0	0	0	0	0	0	1	0	0	0
9	1	134	-0.375	0	0	0	0	0	0	0	0	0	1	0	0
10	1	57	-0.375	0	0	0	0	0	0	0	0	0	0	1	0
11	1	51	-0.375	0	0	0	0	0	0	0	0	0	0	0	1
12	1	73	-0.125	1	0	0	0	0	0	0	0	0	0	0	0
13	1	39	-0.125	0	1	0	0	0	0	0	0	0	0	0	0
14	1	102	-0.125	0	0	1	0	0	0	0	0	0	0	0	0
15	1	283	-0.125	0	0	0	1	0	0	0	0	0	0	0	0
16	1	136	-0.125	0	0	0	0	1	0	0	0	0	0	0	0
17	1	52	-0.125	0	0	0	0	0	1	0	0	0	0	0	0
18	1	85	-0.125	0	0	0	0	0	0	1	0	0	0	0	0
19	1	48	-0.125	0	0	0	0	0	0	0	1	0	0	0	0
20	1	37	-0.125	0	0	0	0	0	0	0	0	1	0	0	0
21	1	102	-0.125	0	0	0	0	0	0	0	0	0	1	0	0
22	1	85	-0.125	0	0	0	0	0	0	0	0	0	0	1	0
23	2	46	-0.375	1	0	0	0	0	0	0	0	0	0	0	0
24	2	59	-0.375	0	1	0	0	0	0	0	0	0	0	0	0
25	2	104	-0.375	0	0	1	0	0	0	0	0	0	0	0	0
26	2	144	-0.375	0	0	0	1	0	0	0	0	0	0	0	0
27	2	167	-0.375	0	0	0	0	1	0	0	0	0	0	0	0
28	2	117	-0.375	0	0	0	0	0	1	0	0	0	0	0	0
29	2	194	-0.375	0	0	0	0	0	0	1	0	0	0	0	0
30	2	148	-0.375	0	0	0	0	0	0	0	1	0	0	0	0
31	2	138	-0.375	0	0	0	0	0	0	0	0	1	0	0	0
32	2	109	-0.375	0	0	0	0	0	0	0	0	0	1	0	0
33	2	65	-0.375	0	0	0	0	0	0	0	0	0	0	1	0
34	2	162	-0.375	0	0	0	0	0	0	0	0	0	0	0	1
35	2	205	-0.125	1	0	0	0	0	0	0	0	0	0	0	0
36	2	182	-0.125	0	1	0	0	0	0	0	0	0	0	0	0
37	2	424	-0.125	0	0	1	0	0	0	0	0	0	0	0	0
38	2	165	-0.125	0	0	0	1	0	0	0	0	0	0	0	0
39	2	192	-0.125	0	0	0	0	1	0	0	0	0	0	0	0
40	2	262	-0.125	0	0	0	0	0	1	0	0	0	0	0	0
41	2	94	-0.125	0	0	0	0	0	0	1	0	0	0	0	0
42	2	112	-0.125	0	0	0	0	0	0	0	1	0	0	0	0
43	2	196	-0.125	0	0	0	0	0	0	0	0	1	0	0	0
44	2	222	-0.125	0	0	0	0	0	0	0	0	0	1	0	0
45	2	169	-0.125	0	0	0	0	0	0	0	0	0	0	1	0
46	3	29	-0.375	1	0	0	0	0	0	0	0	0	0	0	0
47	3	41	-0.375	0	1	0	0	0	0	0	0	0	0	0	0
48	3	88	-0.375	0	0	1	0	0	0	0	0	0	0	0	0
49	3	72	-0.375	0	0	0	1	0	0	0	0	0	0	0
```


##### In10

```
evaluation=Pretreatment(evaluation)
train_month=pd.merge(train_month,product_info_use,on='product_id',how='left')
evaluation=pd.merge(evaluation,product_info_use,on='product_id',how='left')
```






##### In11

```
#测试集和训练集数据
def get_data(table,targets):
    _features = table.drop(targets,axis=1).as_matrix()
    _targets = table[targets].as_matrix()
    _targets.shape = (_targets.shape[0], 1)
    _targets.transpose()
    return _features,_targets
train_month_x,train_month_y = get_data(train_month,'ciiquantity')
evaluation_x,evaluation_y = get_data(evaluation,'ciiquantity_month')
```
## 模型的建立
##### In12

```
#跑模型
rng = np.random.RandomState(1)
clf4=BaggingRegressor(n_estimators=41)
clf4.fit(train_month_x,train_month_y)
predict_month4=clf4.predict(evaluation_x)
print(clf4.score(train_month_x,train_month_y))
answer_table=pd.read_csv('prediction_lilei_20170320.txt')#
print predict_month4
answer_table.ciiquantity_month = predict_month4
answer_table.to_csv('crimp_demo15.txt',index=False)
```

##### Out12

```
0.970191088703
[  74.29268293  187.92682927  221.65853659 ...,   11.2195122     3.95121951
   21.68292683]
```





