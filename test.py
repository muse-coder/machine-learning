# RESTRUCTURE TRAIN DATAFRAME
import numpy as np, pandas as pd, os, gc
train = pd.read_csv('./train.csv')

#pivot是透视表
train2 = train.pivot(index='ImageId', columns='ClassId', values='EncodedPixels')
train2.columns=['e1','e2','e3','e4']#创造列名
#重构dataframe  可以用一个新的train3
train3=pd.DataFrame({'ImageId':list(train2.index),'e1':list(train2['e1']),'e2':list(train2['e2']),'e3':list(train2['e3']),'e4':list(train2['e4'])})
train3.fillna('',inplace=True)#将无效值nan转化为' '（空格）
train3['count'] = np.sum(train3.iloc[:,1:]!='',axis=1).values#对列值数据进行统计
print(train3.head())