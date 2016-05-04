"""
classfication by logistic regression
"""

import os
import numpy as np
import pandas as pd
import time

#caculate denominator
def dlr(x,p):
    return 1+np.exp(p*(theta[0]+np.dot(theta[1:14],x)))

df=pd.read_csv(os.getcwd()+'\wine.txt',dtype='float64',sep=',',names=list(range(14)))

#standardization
df1=df.iloc[:,1:14].apply(lambda x:(x-np.mean(x))/np.std(x))
df=pd.concat([df[0],df1],axis=1)

#logistic regression
dft=df.sample(int(df.shape[0]*0.7))
dfe=(df[~df.index.isin(dft.index)])
dfp=dft.iloc[:,1:14][dft[0]==3]
dfm=dft.iloc[:,1:14][dft[0]!=3]
theta=[0.1]*14
lamb=0.1
alpha=0.1

start=time.time()
for n in range(10):
    for i in range(len(theta)):
        if i==0:
            sp=sum(dfp.apply(lambda x:1/dlr(x,1),axis=1))
            sm=sum(dfm.apply(lambda x:1/dlr(x,-1),axis=1))
        else:
            sp=sum(dfp.apply(lambda x:x[i]/dlr(x,1),axis=1))
            sm=sum(dfm.apply(lambda x:x[i]/dlr(x,-1),axis=1))
        theta[i]=theta[i]+alpha*(sp-sm-lamb*theta[i])

dfe=pd.concat([dfe[0],dfe.iloc[:,1:14].apply(lambda x:np.sign(1/dlr(x,-1)-0.5),axis=1)],axis=1)
dfe.columns=[0,1]
end=time.time()

#result
precise=dfe[(dfe[0]==3)&(dfe[1]==1)].shape[0]/dfe[dfe[1]==1].shape[0]
recall=dfe[(dfe[0]==3)&(dfe[1]==1)].shape[0]/dfe[dfe[0]==3].shape[0]
print(precise,recall)
print(end-start)

