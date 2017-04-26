# coding=utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
import random
import sys
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import xgboost as xgb
from mpl_toolkits.mplot3d import Axes3D
from sklearn.grid_search import  GridSearchCV

plt.rcParams['font.sans-serif']=['SimHei']


finaly_test=pd.read_csv('prediction_lilei_20170320.txt')
product_info=pd.read_csv('product_info.txt')
product_quantity=pd.read_csv('product_quantity.txt')
product_info=product_info.replace(-1,np.nan)
product_quantity=product_quantity.replace(-1,np.nan)
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
product_quantity['product_month']=product_quantity['product_date'].apply(lambda x: x[:7])
month_train=product_quantity.groupby(['product_id','product_month']).sum()['ciiquantity']
month_train=pd.DataFrame(month_train)
month_train=month_train.reset_index()
def regularization(table,name):
    mean, std = table[name].mean(), table[name].std()
    table.loc[:, name] = (table[name] - mean)/std
    return table
def one_hot(table,name):
    dummies = pd.get_dummies(table[name], prefix=name, drop_first=False)
    table = pd.concat([table, dummies], axis=1)
    table = table.drop(name, axis=1)
    return table
product_info_use=product_info.drop(['railway', 'airport', 'citycenter', 'railway2', 'airport2','citycenter2',  'startdate', \
                                    'upgradedate', 'cooperatedate'],axis=1)
# for x in ['eval','eval2','eval3','eval4', 'voters' ,'maxstock']:
#     product_info_use=regularization(product_info_use,x)
def Pretreatment(table):
    table['year']=table['product_month'].apply(lambda x:(float(x[0:4])-2014.0)*100)
    table['month']=table['product_month'].apply(lambda x:x[5:7])
    table=one_hot(table,'month')
    table=table.drop('product_month', axis=1)
    return table
month_train=Pretreatment(month_train)
finaly_test=Pretreatment(finaly_test)
month_train=pd.merge(month_train,product_info_use,on='product_id',how='left')
finaly_test=pd.merge(finaly_test,product_info_use,on='product_id',how='left')


def get_data(table,targets):
    _features = table.drop(targets,axis=1).as_matrix()
    _targets = table[targets].as_matrix()
    _targets.shape = (_targets.shape[0], 1)
    _targets.transpose()
    return _features,_targets
month_train_x,month_train_y = get_data(month_train,'ciiquantity')
finaly_test_x,finaly_test_y = get_data(finaly_test,'ciiquantity_month')


#模型最佳参数的选择
#----------------------------------------------------AdaBoostRegressor-----------------------------------------------------------------------
# params_ada1=np.arange(5,100,5)
# params_ada1=list(params_ada1)
# learning_rata_ada1=[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]
# scores_ada1=[]
# all_score=[]
# rng = np.random.RandomState(5)
# for para in params_ada1:
#     for rate in learning_rata_ada1:
#         clf_ada1=AdaBoostRegressor(n_estimators=para,learning_rate=rate,loss='exponential',random_state=rng)
#         score_ada1=np.sqrt(-cross_val_score(clf_ada1, month_train_x,month_train_y,cv=5,scoring='neg_mean_squared_error'))
#         clf_ada2 = AdaBoostRegressor(n_estimators=para, learning_rate=rate, loss='exponential', random_state=rng)
#         clf_ada2.fit(month_train_x,month_train_y)
#         print 'adabost1:','para:',para,'rate:',rate,'score',np.mean(score_ada1),'accuacy:',clf_ada2.score(month_train_x,month_train_y)
 #para:10 rata:0.1 -->>200.45529 首选
 #para:5  rate:0.05-->>199.92167



#----------------------------------------------------KNeighborsRegressor--------------------------------------------------------------------
# neighbors=np.arange(430,1000,10)
# neighbors=list(neighbors)
# leaf=np.arange(10,100,10)
# leaf=list(leaf)
# for neig in neighbors:
#     for le in leaf:
#         clf_knnreg=KNeighborsRegressor(n_neighbors=neig,algorithm='kd_tree',leaf_size=le)
#         score_knnreg=np.sqrt(-cross_val_score(clf_knnreg, month_train_x,month_train_y,cv=5,scoring='neg_mean_squared_error'))
#         clf_knnreg1 = KNeighborsRegressor(n_neighbors=neig, algorithm='kd_tree', leaf_size=le)
#         clf_knnreg1.fit(month_train_x,month_train_y)
#         print 'knn: ','neighbors:',neig,'leaf:',le,'score:',np.mean(score_knnreg),'accuacy:',clf_knnreg1.score(month_train_x,month_train_y)
# #knn:  neighbors: 140 leaf: 20 score: 213.609246156
#



#----------------------------------------------------DecisionTreeRegressor----------------------------------------------------------------------------------
# rng = np.random.RandomState(1)
# depth=list(np.arange(20,50,1))
# max_fe=list(np.arange(5,27,1))
# for dep in depth:
#     for fe in max_fe:
#         clf_dectree=DecisionTreeRegressor(max_depth=dep,random_state=rng,max_features=fe)
#         score_dectre = np.sqrt(-cross_val_score(clf_dectree, month_train_x, month_train_y, cv=5, scoring='neg_mean_squared_error'))
#         clf_dectree1 = DecisionTreeRegressor(max_depth=dep, random_state=rng, max_features=fe)
#         clf_dectree1.fit(month_train_x,month_train_y)
#         print 'Decsion: ', 'max_depth:', dep, 'max_features:', fe, 'score:', np.mean(score_dectre),'accuacy:',clf_dectree1.score(month_train_x, month_train_y)
# # Decsion:  max_depth: 5 max_features: 17 score: 201.474155529
# # Decsion:  max_depth: 4 max_features: 24 score: 201.183663008



# #----------------------------------------------------ExtraTreesRegressor------------------------------------------------------------------------
# rng = np.random.RandomState(1)
# estimator=list(np.arange(4,60,1))
# depth=list(np.arange(5,30,1))
# for estima in estimator:
#     for dep in depth:
#         clf_extre=ExtraTreesRegressor(n_estimators=estima,max_depth=dep,random_state=rng)
#         score_extre = np.sqrt(-cross_val_score(clf_extre, month_train_x, month_train_y, cv=5, scoring='neg_mean_squared_error'))
#         clf_extre1 = ExtraTreesRegressor(n_estimators=estima, max_depth=dep, random_state=rng)
#         clf_extre1.fit(month_train_x, month_train_y)
#         print 'ExtraTreesRegressor: ', 'n_estimators:', estima, 'max_depth:', dep, 'score:', np.mean(score_extre),\
#             'accuacy:',clf_extre1.score(month_train_x, month_train_y)
#

# #----------------------------------------------------BaggingRegressor--------------------------------------------------------------------------------
# params=np.arange(20,60,1)
# params=list(params)
# scores=[]
# for para in params:
#     clf4_bagg=BaggingRegressor(n_estimators=para)
#     score_bagg = np.sqrt(-cross_val_score(clf4_bagg, month_train_x,month_train_y,cv=5,scoring='neg_mean_squared_error'))
#     clf4_bagg1 = BaggingRegressor(n_estimators=para)
#     clf4_bagg1.fit(month_train_x, month_train_y)
#     print 'BaggingRegressor: ', 'n_estimators:', para,'score:', np.mean(score_bagg),'vaiance:',np.var(score_bagg),\
#         'accuacy:',clf4_bagg1.score(month_train_x,month_train_y)


# ----------------------------------------------------GradientBoostingRegressor-----------------------------------------------------------------------------------uuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu
# for estima in range(10,60,1):
#     for dep in range(30,60,1):
#         clf_gbdtre=GradientBoostingRegressor(n_estimators=estima,max_depth=dep,loss='ls')
#         score_gbd=np.sqrt(-cross_val_score(clf_gbdtre, month_train_x,month_train_y,cv=5,scoring='neg_mean_squared_error'))
#         clf_gbdtre1 = GradientBoostingRegressor(n_estimators=estima, max_depth=dep, loss='quantile')
#         clf_gbdtre1.fit(month_train_x, month_train_y)
#         print 'GradientBoostingRegressor:','n_estimators:', estima, 'max_depth:', dep, 'score:', np.mean(score_gbd),\
#             'accuacy:',clf_gbdtre1.score(month_train_x,month_train_y)


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# #模型
'''
n_estimators=44 48 的结果更好
'''
clf4=BaggingRegressor(n_estimators=30)
rng = np.random.RandomState(1)
clf4=BaggingRegressor(n_estimators=41)
clf4.fit(month_train_x,month_train_y)
predict_month4=clf4.predict(finaly_test_x)
print(clf4.score(month_train_x,month_train_y))
answer_table=pd.read_csv('prediction_lilei_20170320.txt')#
answer_table.ciiquantity_month = predict_month4
answer_table.to_csv('crimp_demo15.txt',index=False)
#
# #不去除iD交叉
clf_bagging = BaggingRegressor(base_estimator=DecisionTreeRegressor(),n_estimators=48)
score_bagging = np.sqrt(-cross_val_score(clf_bagging, month_train_x, month_train_y, cv=5, scoring='neg_mean_squared_error'))
print(np.mean(score_bagging))

#好像有问题
# ----------------------------------------------------XGBRegressor-----------------------------------------------------------------------------------uuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu
# max_depth=list(np.arange(10,40,10))
# n_estimators=list(np.arange(10,200,10))
# xgb_mode=xgb.XGBRegressor()
# clf_xgb=GridSearchCV(xgb_mode,{'max_depth':max_depth,'n_estimators':n_estimators},verbose=1)
# clf_xgb.fit(month_train_x,month_train_y)
# print(clf_xgb.best_score_,clf_xgb.best_params_)