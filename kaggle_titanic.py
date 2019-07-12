import pandas as pd
import numpy as np
import sys

#argument1 = sys.argv[1]
#argument2 = sys.argv[2]
#train = pd.read_csv(argument1)
#test = pd.read_csv(argument2)


###################################################
###############preprocessing#######################
###################################################
#print(train.isnull().sum())
##null값이 있는 feature: Age, Cabin, Embarked

#print(test.isnull().sum())
##null값이 있는 feature: Age, Fare, Cabin

########Parch, SibSp feature#######
## familySize = SibSp + Parch + 1
train['Family'] = train['SibSp'] + train['Parch'] + 1
train['Family'] = train['Family'].astype(int)

test['Family'] = test['SibSp'] + test['Parch'] + 1
test['Family'] = test['Family'].astype(int)

train = train.drop('SibSp', axis = 1)
train = train.drop('Parch', axis = 1)
test = test.drop('SibSp', axis = 1)
test = test.drop('Parch', axis = 1)

##########Ticket feature##########
##ticket number는 전혀 영향X --> just drop it!
train = train.drop('Ticket', axis = 1)
test = test.drop('Ticket', axis = 1)

###########Name feature##########
##이름은 영향X but 이름 중간의 title은 possible! --> extract
title = list(set(train['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())))
#print(title)
train['Title'] = train['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
train = train.drop('Name', axis = 1)
test['Title'] = test['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
test = test.drop('Name', axis = 1)
##mapping // categorical
title_mapping= {
    "Don":        "Noble",
    "Dona":       "Noble",
    "Mr" :        "Mr",
    "Ms":         "Miss",
    "Jonkheer":   "Noble",
    "Dr":         "Officer",
    "Mlle":       "Miss",
    "Mrs" :       "Mrs",
    "Rev":        "Officer",
    "Master" :    "Master",
    "Miss" :      "Miss",
    "Mme":        "Mrs",
    "Major":      "Officer",
    "Lady" :      "Noble",
    "Sir" :       "Noble",
    "Capt":       "Officer",
    "Col":        "Officer",
    "the Countess":"Noble"
}
train.Title = train.Title.map(title_mapping)
test.Title = test.Title.map(title_mapping)

group_train = train.groupby(['Sex', 'Pclass', 'Title'])
train.Age = group_train.Age.apply(lambda x: x.fillna(x.median()))

group_test = test.groupby(['Sex', 'Pclass', 'Title'])
test.Age = group_test.Age.apply(lambda x: x.fillna(x.median()))
## mapping // discretization
title_mapping2 = {
    "Mr":       0,
    "Miss":     1,
    "Mrs":      2,
    "Noble":    3,
    "Officer":  4,
    "Master":   5
}
train.Title = train.Title.map(title_mapping2)
test.Title = test.Title.map(title_mapping2)

########Ebmarked feature#########
## S가 가장많으므로 fillna(S)
train.Embarked = train.Embarked.fillna('S')
test.Embarked = test.Embarked.fillna('S')

embarked_mapping  = {
        "S": 0, 
        "C": 1, 
        "Q": 2
        }
train['Embarked'] = train['Embarked'].map(embarked_mapping)
test['Embarked'] = test['Embarked'].map(embarked_mapping)

#########Cabin feature##########
train['Cabin'] = train['Cabin'].str[:1]
test['Cabin'] = test['Cabin'].str[:1]
Pclass1 = train[train['Pclass']==1]['Cabin'].value_counts()
Pclass2 = train[train['Pclass']==2]['Cabin'].value_counts()
Pclass3 = train[train['Pclass']==3]['Cabin'].value_counts()

#df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
#df.index = ['1st class','2nd class', '3rd class']
#df.plot(kind='bar',stacked=True, figsize=(10,5))
##Cabin  A,B,C only at 1st class

##mapping
cabin_mapping = {
    "A": 0, 
    "B": 0.4, 
    "C": 0.8, 
    "D": 1.2, 
    "E": 1.6, 
    "F": 2, 
    "G": 2.4, 
    "T": 2.8
}
train['Cabin'] = train['Cabin'].map(cabin_mapping)
test['Cabin'] = test['Cabin'].map(cabin_mapping)

train["Cabin"].fillna(train.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
test["Cabin"].fillna(test.groupby("Pclass")["Cabin"].transform("median"), inplace=True)

##########Fare feature##########
test.Fare = test.Fare.fillna(test.Fare.median())

##discretization
train['Fare'].astype(float)
train.loc[(train['Fare'] <= 17), 'Fare'] = 0
train.loc[(train['Fare'] > 17) & (train['Fare'] <= 30), 'Fare'] = 1
train.loc[(train['Fare'] > 30) & (train['Fare'] <= 100), 'Fare'] = 2
train.loc[ train['Fare'] > 100, 'Fare'] = 3

test['Fare'].astype(float)
test.loc[(test['Fare'] <= 17), 'Fare'] = 0
test.loc[(test['Fare'] > 17) & (test['Fare'] <= 30), 'Fare'] = 1
test.loc[(test['Fare'] > 30) & (test['Fare'] <= 100), 'Fare'] = 2
test.loc[ test['Fare'] > 100, 'Fare'] = 3

######PassengerId feature########
PassengerId = test['PassengerId']

train = train.drop('PassengerId', axis = 1)
test = test.drop('PassengerId', axis = 1)

###########Sex feature###########
sex_mapping = {
        "male":   0, 
        "female": 1
        }
train['Sex'] = train['Sex'].map(sex_mapping)
test['Sex'] = test['Sex'].map(sex_mapping)

###########Age feature###########
train['Age'].astype(float)
train.loc[(train['Age'] <= 16), 'Age'] = 0
train.loc[(train['Age'] > 16) & (train['Age'] <= 26), 'Age'] = 1
train.loc[(train['Age'] > 26) & (train['Age'] <= 36), 'Age'] = 2
train.loc[(train['Age'] > 36) & (train['Age'] <= 50), 'Age'] = 3
train.loc[ train['Age'] > 50, 'Age'] = 4

test['Age'].astype(float)
test.loc[(test['Age'] <= 16), 'Age'] = 0
test.loc[(test['Age'] > 16) & (test['Age'] <= 26), 'Age'] = 1
test.loc[(test['Age'] > 26) & (test['Age'] <= 36), 'Age'] = 2
test.loc[(test['Age'] > 36) & (test['Age'] <= 50), 'Age'] = 3
test.loc[ test['Age'] > 50, 'Age'] = 4

###################################################
#########Logistic Regression Classifier############
###################################################

##find w vector
def sigmoid(X, w): #w = [w0, w1, w2, ~]
    z = np.dot(X, w[1:]) + w[0]
    H = 1 / ( 1.0 + np.exp(-z) ) 
    return H

def cost_function(y, hx): 
    m = len(y)
    J = (-1/ m) * sum( y * (np.log(hx)) + (1 - y) * (np.log(1-hx)))
    return  J

def gradient(X, y, w, alpha, iteration):
    
    cost=[]
    for i in range(iteration):
        ##compute the partiatl derivation
        hx = sigmoid(X, w)
        error = hx - y
        grad = X.T.dot(error)
        
        ##update wi
        w[0] = w[0] - alpha*error.sum()
        w[1:] = w[1:] - alpha*grad
        
        cost.append(cost_function(y, hx))
        
    return cost

X = train.iloc[0:, [1,2,3,4,5,6,7,8]].values
y = train.iloc[0:, 0].values

m, n = X.shape
## w -1에서 1사이의 난수발생
w=[]
for i in range(n+1):
    w.append(np.random.uniform(-1,1))
    i+=1

## fix w
#w=[-0.022582230221031452, 0.9834069998091526, -0.32637107375906704,
#   0.4199986658148165, 0.20771198751806574, -0.33167391187489015, 
#   -0.41383528031512573, -0.5667996570726417, 0.17229545163511695]
#print(w)
alpha = 0.000001
iteration = 100000

def LinearRegressionPredict(X):
    return np.where(sigmoid(X, w) >= 0.5, 1, 0)

cost = gradient(X, y, w, alpha, iteration)
#print(cost)

prediction_result = LinearRegressionPredict(X = test.iloc[0:, [0,1,2,3,4,5,6,7]].values)
submission = pd.DataFrame({"PassengerId" : PassengerId, "Survived": prediction_result})
submission.to_csv('1715237.csv', index = False)

sys.exit
