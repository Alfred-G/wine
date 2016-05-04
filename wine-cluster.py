import os
import numpy as np
import pandas as pd
import time

df=pd.read_csv(os.getcwd()+'\wine.txt',sep=',',names=range(14))
df1=df.iloc[:,1:14]
#feature scaling
df1=df1.apply(lambda x:(x-np.mean(x))/np.std(x))
#PCA
val,vec=(np.linalg.eig(np.dot(df1.T,df1)/13))
df1=pd.DataFrame(np.dot(df1,vec.T[:,0:6]))
df=pd.concat([df[0],df1],axis=1)

#K-Mean
kr=df.sample(k)
kr=kr.iloc[:,1:7]
kr.index=range(k)
start=time.time()
for p in range(100):
    df2=df.iloc[:,1:7]
    df=pd.concat([df.iloc[:,0:7],df2.apply(lambda x:kr.apply(lambda y:np.linalg.norm(x-y),1).idxmin(),1).to_frame(14)],axis=1)
    kr=pd.DataFrame([])
    for i in range(k):
        kr=pd.concat([kr,df.iloc[:,1:7][df[14]==i].mean().to_frame(i)],axis=1)
    kr=kr.T
end=time.time()
df.columns=range(8)

print (df[[0,7]].groupby([7,0])[0].count())
print (end-start)
