import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
data_train = pd.read_csv("/Users/gaojie/Kaggle/data/titanic/train.csv")
import matplotlib.pyplot as plt

def set_mising_ages(df):
    age_df=df[['Age','Fare','Parch','SibSp','Pclass']]
    known_age=age_df[age_df.Age.notnull()].values
    unknown_age=age_df[age_df.Age.isnull()].values
    y=known_age[:,0]
    X=known_age[:,1:]
    rfr=RandomForestRegressor(random_state=0,n_estimators=2000,n_jobs=-1)
    rfr.fit(X,y)

    predictedAges=rfr.predict(unknown_age[:,1:])
    df.loc[(df.Age.isnull()),'Age']==predictedAges
    return df,rfr
def set_Cabin_type(df):
    df.loc[(df.Cabin.notnull()),'Cabin']='Yes'
    df.loc[(df.Cabin.isnull()),'Cabin']='No'
    return df
data_train,rfr=set_mising_ages(data_train)
data_train=set_Cabin_type(data_train)



# fig=plt.figure()
# fig.set(alpha=0.5)
# Survived_0=data_train.Survived[data_train.Sex=='female'][data_train.Pclass!=3].value_counts()
# Survived_0.plot(kind='bar')
# # Survived_1=data_train.Sex[data_train.Survived==1].value_counts()
# # df=pd.DataFrame({'survived':Survived_1,'unsurvived':Survived_0})
# plt.title('survived people in gaoji')
# plt.xlabel('survived or not')
# plt.ylabel('number')
