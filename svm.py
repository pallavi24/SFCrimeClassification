import pandas as pd
import numpy as np
import math
import zipfile
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.pipeline import Pipeline



def accuracy(actual, predicted):
    count = 0
    for i in range (0, len(actual)):
        if actual[i]==predicted[i]:
            count = count + 1
    acc = count/len(actual)
    return acc

def llfun(act, pred):
    """ Logloss function for 1/0 probability
    """
    return (-(~(act == pred)).astype(int) * math.log(1e-15)).sum() / len(act)

def readData():
    train = pd.read_csv('data/train/train.csv', parse_dates=['Dates'])[['Dates', 'DayOfWeek', 'PdDistrict', 'Address', 'X', 'Y', 'Category']]
    test = pd.read_csv('data/test/test.csv', parse_dates=['Dates'])
    print("Number of cases in the training set: %s" % len(train))
    print("Number of cases in the testing set: %s" % len(test))
    '''print (train.head(10).to_string())
    print("Testing Data")
    print (test.head(10).to_string())'''
    return (train,test)

def divideIntoTrainAndEvaluationSet(fraction, train):
    msk = np.random.rand(len(train)) < fraction
    trainOnly = train[msk]
    evaluateOnly = train[~msk]
    print("Number of cases in the training only set: %s" % len(trainOnly))
    print("Number of cases in the evaluation  set: %s" % len(evaluateOnly))
    return(trainOnly,evaluateOnly)

def classify(name, train, evaluate, test):
    if(name == "knn"):
        return knnClassifier(train, evaluate, test)
    elif(name =="svm"):
        return svmClassifier(train, evaluate, test)
    elif(name == "logit"):
        return logisticRegressionClassifier(train, evaluate, test)
    elif(name == "dtrees"):
        return dtreesClassifier(train, evaluate, test)
    elif(name == "svm_pipeline"):
        return svm_pipeline(train, evaluate, test)
    else:
        print(" Specify the right name of the classifier : knn/svm/logit/dtrees")

# Move this to a separate function later
'''
    submit = pd.DataFrame({'Id': test.Id.tolist()})
    for category in y.cat.categories:
        submit[category] = np.where(outcomes == category, 1, 0)

    submit.to_csv('k_nearest_neigbour.csv', index = False)
'''

def svmClassifier(train,evaluate, test):
    print('In SVM')
    x = train[['X', 'Y']]
    y = train['Category'].astype('category')
    actual = evaluate['Category'].astype('category')
    svm_classifier = svm.LinearSVC(multi_class='ovr')
    svm_classifier.fit(x, y)
    print('Model fitted..')
    outcomes = svm_classifier.predict(evaluate[['X', 'Y']])
    print('Outcomes predicted.. Calculating log loss')
    val = llfun(actual, outcomes)
    print("Value of LogLoss: " + str(val))
    #print (accuracy(actual, outcomes))
    #return outcomes

    #Testing data
    x_test = test[['X', 'Y']]
    #knn = KNeighborsClassifier(n_neighbors=40)
    svm_classifier.fit(x, y)
    outcomes = svm_classifier.predict(x_test)

    submit = pd.DataFrame({'Id': test.Id.tolist()})
    for category in y.cat.categories:
        submit[category] = np.where(outcomes == category, 1, 0)
    submit.to_csv('svm.csv')
    return outcomes

def logisticRegressionClassifier(train, test):
    print('In Logistic Regression')

def dtreesClassifier(train, test):
    print('In decision trees')

#def createSubmissionFile(lables, fileName):
'''
def svm_pipeline(train,evaluate, test):
    x = train[['X', 'Y']]
    y = train['Category'].astype('category')
    actual = evaluate['Category'].astype('category')
    clf1 = svm.SVC(kernel='linear')
    #clf2 = svm.SVC(kernel='poly')
    classifier = Pipeline([('First', clf1), ('Second', clf2)])
    #################
    classifier.set_params(svc__C=.1).fit(x, y)
    print('Model fitted..')
    outcomes = classifier.predict(evaluate[['X', 'Y']])
    print('Outcomes predicted.. Calculating log loss')
    val = llfun(actual, outcomes)
    print("Value of LogLoss: " + str(val))
    #print (accuracy(actual, outcomes))
    #return outcomes

    #Testing data
    x_test = test[['X', 'Y']]
    #knn = KNeighborsClassifier(n_neighbors=40)
    classifier.fit(x, y)
    outcomes = classifier.predict(x_test)

    submit = pd.DataFrame({'Id': test.Id.tolist()})
    for category in y.cat.categories:
        submit[category] = np.where(outcomes == category, 1, 0)
    submit.to_csv('svm.csv')
    return outcomes
    ######################
'''
def main():
    (train, test) = readData()
    total_train = len(train)
    subset_size = 0.1*total_train
    print("the size of subset "+str(subset_size))
    for_train = train.head(int(subset_size))

    (trainOnly,evaluateOnly) = divideIntoTrainAndEvaluationSet(0.8, for_train)
    print ("Size of trainOnly" +str(len(trainOnly)))
    print ("Size of evaluateOnly" + str(len(evaluateOnly)))

    # Call the classifiers - replace with your classifier
    #predictedLabels = classify("knn",trainOnly, evaluateOnly, test)
    #print (predictedLabels)

    #Classifier SVM
    predictedLabels = classify("svm",trainOnly, evaluateOnly, test)


    print (predictedLabels)

    for i in range(1, len(predictedLabels)):
        print(predictedLabels[i])

    '''
    y = train['Category'].astype('category')
    for i in range(1, len(y)):
        print(y[i])
    #print(accuracy(evaluateOnly['Categories'], predictedLabels))
    '''

main()
