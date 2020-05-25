import numpy as np
import scipy
import pandas as pd
import seaborn as sns
import  matplotlib.pyplot as plt

def main():
    train_frame=pd.read_csv('data_files/train.csv')
    print(train_frame)
    test_frame=pd.read_csv('data_files/test.csv')


    #sns.distplot(train_frame[train_frame['Survived']==1][['Age']])
    #sns.distplot(train_frame[(train_frame['Survived']==1) & (train_frame['Sex']=='female')][['Age']], bins=30)
    #sns.distplot(train_frame[(train_frame['Survived']==1) & (train_frame['Sex']=='male')][['Age']], bins= 30)
    #plt.show()
    #train_frame=train_frame.drop(['PassengerId'],axis=1)
    train_frame['Cabin'].fillna("U0",inplace=True)
    test_frame['Cabin'].fillna("U0",inplace=True)
    print(train_frame)
    train_frame['Deck']=train_frame['Cabin'].map(lambda x:ord(x[0][0])-ord('A'))
    test_frame['Deck']=test_frame['Cabin'].map(lambda x:ord(x[0][0])-ord('A'))
    print(train_frame)
    #sns.barplot(x='Deck',y='Survived',data=train_frame)
    train_frame['Deck'].hist(bins=120)
    #plt.show()
    train_frame.drop('Cabin',inplace=True,axis=1)
    mean_age=train_frame['Age'].mean()
    mean_age2=test_frame['Age'].mean()

    train_frame['Age'].fillna(value=mean_age,inplace=True)
    test_frame['Age'].fillna(value=mean_age2,inplace=True)
    train_frame['Embarked'].fillna('S',inplace=True)
    test_frame['Embarked'].fillna('S',inplace=True)
    train_frame[['Age','Fare']]=train_frame[['Age','Fare']].astype(int)
    test_frame[['Age','Fare']]=test_frame[['Age','Fare']].astype(int)
    print(train_frame.info())
    genders={'male':0,'female':1}
    train_frame['Sex']=train_frame['Sex'].map(genders)
    test_frame['Sex']=test_frame['Sex'].map(genders)
    print(train_frame['Ticket'].describe())
    train_frame.drop('Ticket',axis=1,inplace=True)
    test_frame.drop('Ticket',axis=1,inplace=True)
    embarked={'Q':0,'S':1,'C':2}
    train_frame['Embarked']=train_frame['Embarked'].map(embarked)
    test_frame['Embarked']=test_frame['Embarked'].map(embarked)
    train_frame['Age']=(pd.cut(train_frame['Age'],bins=[0,17,25,28,32,40,100],labels=[0,1,2,3,4,5],include_lowest=True))
    test_frame['Age']=(pd.cut(test_frame['Age'],bins=[0,17,25,28,32,40,100],labels=[0,1,2,3,4,5],include_lowest=True))
    train_frame[['Fare']]=pd.qcut(train_frame['Fare'],4)
    test_frame[['Fare']]=pd.qcut(test_frame['Fare'],4)
    #print(train_frame['Fare'].value_counts())

main()