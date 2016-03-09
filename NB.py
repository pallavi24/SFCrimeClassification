import pandas as pd
import numpy as np
import math
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder

#Load data from csv
def readData():
    train = pd.read_csv('data/train/train.csv', parse_dates=['Dates'])[['Dates', 'DayOfWeek', 'PdDistrict', 'X', 'Y', 'Category']]
    test = pd.read_csv('data/test/test.csv', parse_dates=['Dates'])
    print("Number of cases in the training set: %s" % len(train))
    print("Number of cases in the testing set: %s" % len(test))
    return (train,test)

#Feature extraction
def applyFunction(train, inputCol, check, outputCol):
    train[outputCol] = train[inputCol].apply(lambda x:1 if x == check else 0 )
    return train

def convertToFeatures(train):
    train = applyFunction(train, 'DayOfWeek',"Sunday", "sun")
    train = applyFunction(train, 'DayOfWeek',"Monday", "mon")
    train = applyFunction(train, 'DayOfWeek', "Tuesday", "tues")
    train = applyFunction(train, 'DayOfWeek',"Wednesday", "wed")
    train = applyFunction(train, 'DayOfWeek',"Thursday", "thur")
    train = applyFunction(train, 'DayOfWeek',"Friday", "fri")
    train = applyFunction(train, 'DayOfWeek',"Saturday", "sat")
    train = applyFunction(train, 'PdDistrict',"BAYVIEW", "BAYVIEW")
    train = applyFunction(train, 'PdDistrict',"CENTRAL", "CENTRAL")
    train = applyFunction(train, 'PdDistrict',"INGLESIDE", "INGLESIDE")
    train = applyFunction(train, 'PdDistrict',"MISSION", "MISSION")
    train = applyFunction(train, 'PdDistrict',"NORTHERN", "NORTHERN")
    train = applyFunction(train, 'PdDistrict',"PARK", "PARK")
    train = applyFunction(train, 'PdDistrict',"RICHMOND", "RICHMOND")
    train = applyFunction(train, 'PdDistrict',"SOUTHERN", "SOUTHERN")
    train = applyFunction(train, 'PdDistrict',"TARAVAL", "TARAVAL")
    train = applyFunction(train, 'PdDistrict',"TENDERLOIN", "TENDERLOIN")
    return train




#Subsetting data
def sliceByCategory(categories, train):
    trainWithCategories = train.loc[train['Category'].isin(categories)]
    return trainWithCategories

#Partition Train data for evaluating
def divideIntoTrainAndEvaluationSet(fraction, train):
    msk = np.random.rand(len(train)) < fraction
    trainOnly = train[msk]
    evaluateOnly = train[~msk]
    print("Number of cases in the training only set: %s" % len(trainOnly))
    print("Number of cases in the evaluation  set: %s" % len(evaluateOnly))
    return(trainOnly,evaluateOnly)

#Calculating LogLoss
def llfun(act, pred):
    """ Logloss function for 1/0 probability
    """
    return (-(~(act == pred)).astype(int) * math.log(1e-15)).sum() / len(act)

#Classifify using Naive Bayes
def NaiveBayesClassifier(train, evaluate, test):
    x = train[['X', 'Y', 'mon', 'sun', 'tues', 'thur', 'fri', 'sat', 'BAYVIEW',
               'CENTRAL', 'INGLESIDE', 'MISSION', 'NORTHERN', 'PARK', 'RICHMOND', 'SOUTHERN', 'TARAVAL', 'TENDERLOIN']]
    #print(x[])
    y = train['Category'].astype('category')
    #print(pd.unique(y.ravel()))
    #all_categories =pd.unique(y.ravel())
    actual = evaluate['Category'].astype('category')
    classifier = GaussianNB()

    x_test = evaluate[['X', 'Y', 'mon', 'sun', 'tues', 'thur', 'fri', 'sat', 'BAYVIEW',
               'CENTRAL', 'INGLESIDE', 'MISSION', 'NORTHERN', 'PARK', 'RICHMOND', 'SOUTHERN', 'TARAVAL', 'TENDERLOIN']]
    #y_pred =
    classifier.fit(x,y)#.predict(evaluate[['X', 'Y', 'DayOfWeek']])
    print(classifier)
    y_pred = classifier.predict(x_test)
    y_prob = classifier.predict_proba(x_test)
    val = llfun(actual, y_pred)
    print('Log loss: ' + str(val))

    #On testing data
    #x_test = test[['X', 'Y', 'mon', 'sun', 'tues', 'thur', 'fri', 'sat', 'BAYVIEW',
    #          'CENTRAL', 'INGLESIDE', 'MISSION', 'NORTHERN', 'PARK', 'RICHMOND', 'SOUTHERN', 'TARAVAL', 'TENDERLOIN']]
    #x_test = x_test.head(10)
    #assert isinstance(x_test, object)
    #test_outcomes = classifier.predict(x_test)
    #submit = pd.DataFrame({'Id': test.Id.tolist()})
    #for category in all_categories:
    #    submit[category] = np.where(y_pred == category, 1, 0)

    #submit.to_csv('NB_test.csv', index = False)
    return y_pred

def main():
    (train, test) = readData()
    train = convertToFeatures(train)
    test = convertToFeatures(test)

    #Subsetting
    categories = ["LARCENY/THEFT", "OTHER OFFENSES", "NON-CRIMINAL","ASSAULT", "DRUG/NARCOTIC"]
    trainWithTopCategories = sliceByCategory(categories, train)
    print (trainWithTopCategories.loc[1:10])
    #print(test.columns.values)
    (trainOnly,evaluateOnly) = divideIntoTrainAndEvaluationSet(0.8, trainWithTopCategories)
    predictedLabels = NaiveBayesClassifier(trainOnly, evaluateOnly, test)
    #print (predictedLabels)

main()
