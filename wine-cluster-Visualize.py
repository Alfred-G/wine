"""
visualize features
and save the picture.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv(os.getcwd()+'\wine.txt',sep=',',names=range(14))
df1=df.iloc[:,1:14].apply(lambda x:(x-np.mean(x))/np.std(x))
df=pd.concat([df[0],df1],axis=1)

d={1:'b',2:'g',3:'r'}
for n in range(13):
    for i in range(1,4):
        df1=df[df[0]==i]
        plt.scatter(df1[n+1],df1[0]+n*5,c=d[i])

plt.savefig(os.getcwd()+'/feature.png')
plt.show()
