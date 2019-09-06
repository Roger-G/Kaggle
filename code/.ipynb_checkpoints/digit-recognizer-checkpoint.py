import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn import neighbors
from pandas import DataFrame

df_train =pd.read_csv("/Users/gaojie/Kaggle/data/digit-recognizer/train.csv")
df_test =pd.read_csv("/Users/gaojie/Kaggle/data/digit-recognizer/test.csv")
train_labels=df_train.label
knn=neighbors.KNeighborsClassifier(algorithm='kd_tree',n_neighbors=3)
print("training")
train_data = df_train.drop(columns='label')
knn.fit(train_data,train_labels)
print('get test data')
predictions=knn.predict(df_test)

s={'ImageId':list(range(1,len(df_test)))}
s=DataFrame(s)
result = pd.DataFrame({'PassengerId':df_test.indx.values,'Label':predictions.astype(np.int32)})
result.to_csv("/Users/gaojie/Kaggle/data/digit-recognizer/nn_predictions1.csv",index=False)
print('finished!')