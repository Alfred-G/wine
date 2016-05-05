import os
import numpy as np
import pandas as pd
import time

df=pd.read_csv(os.getcwd()+'\wine.txt',dtype='float64',sep=',',names=list(range(14)))
df1=df.iloc[:,1:14]

#PCA
df1=df1.apply(lambda x:(x-np.mean(x))/np.std(x))
val,vec=(np.linalg.eig(np.dot(df1.T,df1)/13)) 
df1=pd.DataFrame(np.dot(df1,vec.T[:,0:6])) 

df=pd.concat([df[0],df1],axis=1)
df.columns=range(7)
dfl=df[0]
dfl[dfl!=1]=-1
df=df.iloc[:,1:7]
dft=df.sample(int(df.shape[0]*0.7))
dftl=dfl[dfl.index.isin(dft.index)]
dfe=(df[~df.index.isin(dft.index)])
dfel=(dfl[~dfl.index.isin(dft.index)])

theta=np.array([0.1]*int(df.shape[0]*0.7))
alpha=0.1

#SVM
start=time.time()
for n in range(1000):
    partial=1-dftl*dft.apply(lambda x:np.dot(theta,dftl*dft.apply(lambda y:np.dot(x,y),axis=1)),axis=1)
    theta=theta+partial
    w=np.dot(theta,pd.DataFrame.multiply(dft,dftl,axis=0))

#result
dfe=np.dot(dfe,w)
dfel=pd.DataFrame({0:dfel,1:np.sign(dfe)})
end=time.time()
precise=dfel[(dfel[0]==1)&(dfel[1]==1)].shape[0]/dfel[dfel[1]==1].shape[0]
recall=dfel[(dfel[0]==1)&(dfel[1]==1)].shape[0]/dfel[dfel[0]==1].shape[0]
print(precise,recall)
print('Time: %fs'%(end-start))
