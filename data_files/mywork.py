import numpy as np
import scipy
import sys
import pandas as pd
import seaborn as sns
import  matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import cross_val_score
from sklearn.linear_model  import  LogisticRegression
from sklearn import  linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC,LinearSVC
def main():
    "dictionaries"
    mf={'male': 0, 'female': 1}
    emb={'Q':0,'S':1,'C':2}
    "dataframe reading"
    tst=pd.read_csv('test.csv')
    trn=pd.read_csv('train.csv')
    names=pd.read_csv('test.csv')
    "graph analysis"
    "columns"
    print(tst.columns.values)
    #['PassengerId' 'Pclass' 'Name' 'Sex' 'Age' 'SibSp' 'Parch' 'Ticket' 'Fare'
    #'Cabin' 'Embarked']
    #probable cols= Age,Sex,Embarked,Cabin,Pclass,

    #plot analysis
    # male survived-100+/577
    #female survived-250/314
    #middle age survival more;
    #print(trn[trn['Sex']=='female'].count())
    #trn[ trn['Survived']==1]['Sex'].hist()
    #trn['Embarked'].hist()
    #plt.show()



    "missing analysis"
    #we might consider dropping cabins
    #print(trn.isnull().sum())
    mean_age=trn['Age'].mean()
    trn['Age'].fillna(mean_age,inplace=True)
    tst['Age'].fillna(mean_age,inplace=True)
    trn['Embarked'].fillna('S',inplace=True)
    tst['Embarked'].fillna('S',inplace=True)

    trn.drop('Cabin',inplace=True,axis=1)
    tst.drop('Cabin',inplace=True,axis=1)
    #print(trn.isnull().sum())
    trn.drop('Name',inplace=True,axis=1)
    tst.drop('Name',inplace=True,axis=1)
    trn.drop('Ticket',inplace=True,axis=1)
    tst.drop('Ticket',inplace=True,axis=1)
    trn.drop('PassengerId',inplace=True,axis=1)
    tst.drop('PassengerId',inplace=True,axis=1)



    #trn[trn['Survived']==1].hist()
    #plt.show()

    trn[['Age','Fare']]=trn[['Age','Fare']].astype(int)
    tst[['Age','Fare']]=trn[['Age','Fare']].astype(int)

    "feature engineering"
    #print(tst.isnull().sum())

    data=[trn,tst]
    for dataset in data:
        dataset['Age'] = dataset['Age'].astype(int)
        dataset.loc[dataset['Age'] <= 11, 'Age'] = 0
        dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3
        dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 30), 'Age'] = 4
        dataset.loc[(dataset['Age'] > 30) & (dataset['Age'] <= 40), 'Age'] = 5
        dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6
        dataset.loc[dataset['Age'] > 66, 'Age'] = 6
    #print(trn['Age'].value_counts())

    trn['Fare']=pd.qcut(trn['Fare'],q=6,labels=[0,1,2,3,4,5]).astype(int)
    tst['Fare']=pd.qcut(tst['Fare'],q=6,labels=[0,1,2,3,4,5]).astype(int)
    #print(trn.head())
    #print(trn.dtypes)
    print(tst.head())
    trn['Sex'] = trn['Sex'].map(mf)
    tst['Sex'] = tst['Sex'].map(mf)
    trn['Embarked'] = trn['Embarked'].map(emb)
    tst['Embarked'] = tst['Embarked'].map(emb)
    tst['Sex'].fillna(tst['Sex'].mode(),inplace=True)
    tst['Embarked'].fillna(tst['Embarked'].mode(),inplace=True)
    print(tst.head())


    #model application

    x_train=trn.drop('Survived',axis=1)
    y_train=trn['Survived']
    x_test=tst
    print(x_test.columns.values)
    print(x_train.columns.values)



    #data_ready

    #1.sgd classifier-sadi bhasha me bolo to gradient descent
    sgd=linear_model.SGDClassifier(max_iter=5,tol=None)
    sgd.fit(x_train,y_train)
    #print(x_test.isnull().sum())
    #print(x_train.isnull().sum())
    #y_pred=sgd.predict(x_test)
    print(sgd.score(x_train,y_train))
    #accuracy=76 percent

    #2 random classifier
    randomforest=RandomForestClassifier(n_estimators=150,oob_score=True)
    randomforest.fit(x_train,y_train)
    print(randomforest.score(x_train,y_train))
    y_pred=randomforest.predict(x_test)
    #accuracy=91.7 percent

    #logistic regression classifier
    logreg=LogisticRegression()
    logreg.fit(x_train,y_train)
    print(logreg.score(x_train,y_train))
    #accuracy=80.13 percent

    #k nearest neighbour
    knearest=KNeighborsClassifier()
    knearest.fit(x_train,y_train)
    print(knearest.score(x_train,y_train))
    #accuracy=85.7 percent

    #Gaussian Naive Bayes
    Gnb=GaussianNB()
    Gnb.fit(x_train,y_train)
    print(Gnb.score(x_train,y_train))
    #accuracy=76.5 percent

    #Perceptron
    prcpt=Perceptron(max_iter=5)
    prcpt.fit(x_train,y_train)
    print(prcpt.score(x_train,y_train))
    #accuracy=77.1 percent

    #linear svm
    LSVC=LinearSVC()
    LSVC.fit(x_train,y_train)
    print(LSVC.score(x_train,y_train))

    #accuracy~80 percent



    "Clearly Random forest classifier seems to promise highest accuracy,so now cross validation"
    scores=cross_val_score(randomforest,x_train,y_train,cv=10,scoring='accuracy')
    print(scores)
    print(scores.mean())
    print(scores.std())

    #[0.76666667 0.80898876 0.76404494 0.85393258 0.86516854 0.84269663
    #0.84269663 0.80898876 0.84269663 0.82022472]
    #0.8216104868913858-mean
    #0.033059112756476425-std
    #seems good huh

    print("oob score:", round(randomforest.oob_score_, 4)*100, "%")
    y_pred=pd.DataFrame(y_pred)
    y_pred.columns=['Survived']
    y_pred['PassengerId']=(names['PassengerId'])
    print(y_pred.head())
    y_pred=y_pred[['PassengerId','Survived']]
    print(y_pred.head())
    y_pred.to_csv('submission.csv',index=False)
















main()



