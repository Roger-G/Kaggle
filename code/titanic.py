import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.impute import SimpleImputer
import datetime
import warnings
from sklearn.exceptions import DataConversionWarning
import seaborn as sns
from sklearn.model_selection import cross_val_score,train_test_split,learning_curve
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingRegressor
import multiprocessing
multiprocessing.set_start_method('spawn', True)
# Suppress warnings
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)
###############################################################################
data_train = pd.read_csv("/Users/gaojie/Kaggle/data/titanic/train.csv")
data_test = pd.read_csv("/Users/gaojie/Kaggle/data/titanic/test.csv")
# Data exploration
combine=[data_train,data_test]

fig1=plt.figure()
data_train.Age[data_train.Survived==1].plot(kind='kde')
data_train.Age[data_train.Survived==0].plot(kind='kde')
plt.xlabel('age')
plt.ylabel('number of people')
plt.legend(('survivors','no-survivors'),loc='best')
g=sns.FacetGrid(data_train,col='Survived')
g.map(plt.hist,'Age',alpha=.5,bins=30)
g.add_legend()
gr=sns.FacetGrid(data_train,col='Survived')
gr.map(plt.hist,'Fare',bins=30)

# data_train.Pclass[data_train.Survived==1]
g2=sns.FacetGrid(data_train,col='Pclass')
g2.map(plt.hist,'Fare',bins=10)
# plt.show()
# plt.draw()
# plt.pause(0.001)
for dataset in combine:
    dataset['Title']=dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    dataset['Title']=dataset['Title'].replace(['Lady','Countness','Capt','Col',\
        'Don','Dr','Major','Rev','Sir','Jonkheer','Dona'],'Rare')
    dataset['Title']=dataset['Title'].replace(['Miss','Mlle','Mrs','Ms'],'Female_name')
    dataset['Title']=dataset['Title'].replace(['Mr','Sir'],'male_name')
fig11=plt.figure()
data_train[['Title','Survived']].groupby(['Title']).mean().plot(kind='bar')
Title_mapping={'Rare':1,'Female_name':2,'male_name':3}
for dataset in combine:
    dataset['Title']=dataset['Title'].map(Title_mapping)
    dataset['Title']=dataset['Title'].fillna(0)
# print(data_train.Title[data_train['Survived']==1].value_counts())
# plt.show()

data_train[['Pclass','Survived']].groupby(['Pclass']).mean().plot(kind='bar')
# fig4=plt.figure()
a=data_train.Fare[data_train.Survived==1].mean()
b=data_train.Fare[data_train.Survived==0].mean()
pd.DataFrame([a,b]).plot(kind='bar')
# fig5=plt.figure()
# plt.show()
f,ax=plt.subplots(1,2,figsize=(18,8))
data_train[['Parch','Survived']].groupby(['Parch']).mean().plot.bar(ax=ax[0])
ax[0].set_title('Parch and Survived')
data_train[['SibSp','Survived']].groupby(['SibSp']).mean().plot.bar(ax=ax[1])
ax[1].set_title('SibSp and Survived')

fig4=plt.figure()
data_train['Familysize']=data_train['SibSp']+data_train['Parch']+1
data_train[['Familysize','Survived']].groupby(['Familysize']).mean().plot(kind='bar')
# plt.show()
for dataset in combine:
    dataset['Familysize']=dataset['SibSp']+dataset['Parch']+1
    dataset['Goodfamilysize']=1
    dataset.Goodfamilysize[dataset['Familysize']==1]=2
    dataset.Goodfamilysize[dataset['Familysize']==2]=3
    dataset.Goodfamilysize[dataset['Familysize']==3]=3
    dataset.Goodfamilysize[dataset['Familysize']==4]=3


###############################################################################
# feature enginering
def set_mising_ages(df):
    age_df=df[['Age','Fare','Parch','SibSp','Pclass']]
    known_age=age_df[age_df.Age.notnull()].values
    unknown_age=age_df[age_df.Age.isnull()].values
    y=known_age[:,0]
    X=known_age[:,1:]
    rfr=RandomForestRegressor(random_state=0,n_estimators=2000,n_jobs=-1)
    rfr.fit(X,y)
    predictedAges=rfr.predict(unknown_age[:,1:])
    df.loc[(df.Age.isnull()),'Age']=predictedAges
    # df.loc[(df.Age.isnull()),'Age']=df.Age.median()
    return df,rfr
def set_mising_Fare(df):
    fare_df=df[['Fare','Age','Parch','SibSp','Pclass']]
    known_fare=fare_df[fare_df.Fare.notnull()].values
    unknown_fare=fare_df[fare_df.Fare.isnull()].values
    y=known_fare[:,0]
    X=known_fare[:,1:]
    rfr=RandomForestRegressor(random_state=1,n_estimators=2000,n_jobs=-1)
    rfr.fit(X,y)
    predictedfare=rfr.predict(unknown_fare[:,1:])
    df.loc[(df.Fare.isnull()),'Fare']=predictedfare
    # df.loc[(df.Fare.isnull()),'Fare']=df.Fare.median()
    return df
def set_Cabin_type(df):
    df.loc[(df.Cabin.notnull()),'Cabin']='Yes'
    df.loc[(df.Cabin.isnull()),'Cabin']='No'
    return df
data_train,rfr=set_mising_ages(data_train)
data_train=set_Cabin_type(data_train)
# dummies_Cabin=pd.get_dummies(data_train['Cabin'],prefix='Cabin')
dummies_Embarked = pd.get_dummies(data_train['Embarked'],prefix='Embarked')
dummies_Sex=pd.get_dummies(data_train['Sex'],prefix='Sex')
dummies_Pclass= pd.get_dummies(data_train['Pclass'],prefix='Pclass')
# data_train.Sex[data_train.Sex=='female']=1
# data_train.Sex[data_train.Sex=='male']=0


data_train.loc[ data_train['Age'] <= 16, 'Age'] = 0
# data_train.loc[(data_train['Age'] > 4) & (data_train['Age'] <= 14), 'Age'] = 1
data_train.loc[(data_train['Age'] > 16) & (data_train['Age'] <= 33), 'Age'] = 2
data_train.loc[(data_train['Age'] > 33) & (data_train['Age'] <= 57), 'Age'] = 3
data_train.loc[ data_train['Age'] > 57, 'Age']=4
# data_train
df_train=pd.concat([data_train,dummies_Embarked,dummies_Sex,dummies_Pclass],axis=1)
df_train['pclass_sex']=df_train['Pclass'] * df_train['Sex_male']
df_train.drop(['Pclass','Name','Sex','Ticket','Cabin','Embarked'],axis=1,inplace=True)


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
# scaler.fit(df_train['Age'].values.reshape(-1,1))
# # scaler
# df_train['Age_scale']=scaler.transform(df_train['Age'].values.reshape(-1,1))
# df_train
scaler.fit(df_train['Fare'].values.reshape(-1,1))
# scaler
df_train['Fare_scale']=scaler.transform(df_train['Fare'].values.reshape(-1,1))
df_train.drop(['Fare'],axis=1,inplace=True)

train_df = df_train.filter(regex='Survived|Title|Goodfamilysize|Age|Parch|Fare_.*|Sex_.*|Pclass_.*|pclass_sex')
train_np = train_df.values
X=train_np[:,1:]

y=train_np[:,0]
###############################################################################
# Build a Model 

clf=linear_model.LogisticRegression(C=1.0,penalty='l1',tol=1e-6,solver='liblinear')
clf.fit(X,y)
print (cross_val_score(clf, X, y, cv=5))

# check the cofficient and feature to find different 
print(pd.DataFrame({"columns":list(train_df.columns)[1:], "coef":list(clf.coef_.T)}))

###############################################################################
###############################################################################
# Testdata preprocessing

# data_test.loc[ (data_test.Fare.isnull()), 'Fare' ] = 0

tmp_df = data_test[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
null_age = tmp_df[data_test.Age.isnull()].values
# 根据特征属性X预测年龄并补上
X_age = null_age[:, 1:]

predictedAges = rfr.predict(X_age)
data_test.loc[ (data_test.Age.isnull()), 'Age' ] = predictedAges
data_test=set_mising_Fare(data_test)
data_test.loc[ data_test['Age'] <= 16, 'Age'] = 0
data_test.loc[(data_test['Age'] > 16) & (data_test['Age'] <= 33), 'Age'] = 1
data_test.loc[(data_test['Age'] > 33) & (data_test['Age'] <= 57), 'Age'] = 2
# data_test.loc[(data_test['Age'] > ) & (data_test['Age'] <= 57), 'Age'] = 3
data_test.loc[ data_test['Age'] > 57, 'Age']


data_test=set_Cabin_type(data_test)
dummies_Cabin=pd.get_dummies(data_test['Cabin'],prefix='Cabin')
dummies_Embarked = pd.get_dummies(data_test['Embarked'],prefix='Embarked')
dummies_Sex=pd.get_dummies(data_test['Sex'],prefix='Sex')
dummies_Pclass= pd.get_dummies(data_test['Pclass'],prefix='Pclass')


# dummies_Cabin=pd.get_dummies(data_test['Cabin'],prefix='Cabin')
# 如何concat 
df_test=pd.concat([data_test,dummies_Embarked,dummies_Cabin,dummies_Sex,dummies_Pclass],sort=False,axis=1)
df_test['pclass_sex']=df_test['Pclass']*df_test['Sex_male']

df_test.drop(['Pclass','Name','Sex','Cabin','Embarked','Ticket'],axis=1,inplace=True)

# df_test=df_test[df_test.Age>18]
# df_test['Age_scaled']=scaler.transform(df_test['Age'].values.reshape(-1,1))
df_test['Fare_scaled']=scaler.transform(df_test['Fare'].values.reshape(-1,1))
# df_test.loc[ (df_test.Fare_scaled.isnull()), 'Fare_scaled' ] = 0
df_test.drop(['Fare'],axis=1,inplace=True)

test=df_test.filter(regex='Title|Age|Goodfamilysize|Parch|Fare_.*|Sex_.*|Pclass_.*|pclass_sex')
############################################# bagging
# clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
# bagging_clf=BaggingRegressor(base_estimator=linear_model.LogisticRegression(),n_estimators=10,bootstrap=True, bootstrap_features=False)
# bagging_clf.fit(X,y)
# predictions= bagging_clf.predict(test)
# result = pd.DataFrame({'PassengerId':data_test['PassengerId'].values,'Survived':predictions.astype(np.int32)})
# result.to_csv("/Users/gaojie/Kaggle/data/titanic/logistic_regression_predictions12.csv",index=False)

# pd.DataFrame({'columns':list(train_df.columns)[1:],'coef':list(clf.coef_.T)})
####no bagging

predictions= clf.predict(test)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].values,'Survived':predictions.astype(np.int32)})
result.to_csv("/Users/gaojie/Kaggle/data/titanic/logistic_regression_predictions19.csv",index=False)

# test.shape (891,14)
# test.fillna(test.mean(),inplace=True)
###############################################################################
#Make prediction


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, 
                        train_sizes=np.linspace(.05, 1., 20), verbose=0, plot=True):
    train_sizes, train_scores, test_scores = learning_curve(estimator, 
    X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    if plot:
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel(u"training data")
        plt.ylabel(u"score")
        # plt.gca().invert_yaxis()
        plt.grid()
    
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, 
                         alpha=0.1, color="b")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, 
                         alpha=0.1, color="r")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label="training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label="test score")
    
        plt.legend(loc="best")
        plt.show()
        # plt.draw()
        # plt.pause(0.001)   
        plt.gca().invert_yaxis()
    
    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
    return midpoint, diff

plot_learning_curve(clf,'learning curve', X, y)


###############################################################################
#Model ensemble
