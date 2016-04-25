import os
import pandas as pd
from sklearn import tree

df=pd.read_csv(os.getcwd()+'\wine.txt',dtype='float64',sep=',',names=list(range(14)))
ac=0
for i in range(10):
    dft=df.sample(int(df.shape[0]*0.7))
    dfe=(df[~df.index.isin(dft.index)])
    model = tree.DecisionTreeClassifier(criterion='gini')
    model.fit(dft.iloc[:,1:14], dft[0])
    predicted= model.predict(dfe.iloc[:,1:14])
    ac+=sum(predicted==dfe.iloc[:,0])/len(predicted)
print (ac/10)
