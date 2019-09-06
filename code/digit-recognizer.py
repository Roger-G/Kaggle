import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn import neighbors
from pandas import DataFrame
import sklearn
from collections import Counter
from sklearn import preprocessing
from sklearn.decomposition import PCA
import datetime
import seaborn as sns
from sklearn.model_selection import cross_val_score,train_test_split
df_train =pd.read_csv("/Users/gaojie/Kaggle/data/digit-recognizer/train.csv")
df_test =pd.read_csv("/Users/gaojie/Kaggle/data/digit-recognizer/test.csv")
train_labels=df_train.label



train_data = df_train
# train_scale=preprocessing.scale(train_data)
print(Counter(train_data.label))
sns.countplot(train_data.label)

split_train,split_cv=train_test_split(train_data,test_size=0.3,random_state=42)
X_train=split_train.values[:,1:]
y_train=split_train.values[:,0]
plt.figure(figsize=(12,6))
x,y=10,4

# turn to gray picture
X_train[X_train<=127]=0
X_train[X_train>127]=1
# for i in range(40):
#     plt.subplot(y,x,i+1)
#     plt.imshow(X_train[i].reshape((28,28)),interpolation='nearest')
# plt.show()

# implement PCA 
# pca= PCA(n_components=20)
# pca.fit()

knn=neighbors.KNeighborsClassifier(algorithm='kd_tree',n_neighbors=3)
print("training")
knn.fit(X_train,y_train)
print('get test data')
starttime = datetime.datetime.now()
X_val=split_cv.values[:,1:]
y_val=split_cv.values[:,0]
X_val[X_val<=127]=0
X_val[X_val>127]=1
print(" start to do cv")
print(cross_val_score(knn,X_val,y_val,cv=5))

#long running
#do something other
endtime = datetime.datetime.now()
print ((endtime - starttime).seconds)

# s={'ImageId':list(range(1,len(df_test)))}
# s=DataFrame(s)
# result = pd.DataFrame({'PassengerId':s.ImageId,'Label':predictions.astype(np.int32)})
# result.to_csv("/Users/gaojie/Kaggle/data/digit-recognizer/nn_predictions1.csv",index=False)
print('finished!')