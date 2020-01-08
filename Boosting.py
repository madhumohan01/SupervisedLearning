from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from helpers import readDataFromCSV
from helpers import readDataFromFile
import datetime
import numpy as np
import matplotlib.pyplot as plt

print("Starting processing Boosting...")
for datasetid in range(2):
    if datasetid == 0:
        X, y = readDataFromCSV('adult.csv')
        dataset = 'ADULT'
        feature_names = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country']
        # X, y = readDataFromFile('winequality-white.csv')
    else:
        X, y = readDataFromFile('winequality-white.csv')
        feature_names = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']
        dataset = 'WINE'
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    results = []
    # for learning_rate in [1.0,1.2,1.5]:
    for learning_rate in [1.0,1.5,2.0]:
        result = []
        for estimators in [50,100,150,200,250]:
        # for estimators in [50,100]:
            print("Processing for learning_rate", learning_rate, " and estimators", estimators)
            bdt = AdaBoostClassifier(
                DecisionTreeClassifier(max_depth=7),
                n_estimators=estimators,
                learning_rate=learning_rate)
            start = datetime.datetime.now()
            bdt.fit(X_train, y_train)
            finish = datetime.datetime.now()
            # print(estimators, learning_rate, bdt.score(X_train, y_train),bdt.score(X_test, y_test),(finish-start).total_seconds())
            result.append([learning_rate, estimators, bdt.score(X_train, y_train),bdt.score(X_test, y_test),(finish-start).total_seconds()])
        result = np.asarray(result)
        results.append(result)
    results = np.asarray(results)
    # print(results)
    ax = plt.axes()
    ax.plot(results[0,:,1], results[0,:,3], label='Score @ learning rate 1.0')
    ax.plot(results[1,:,1], results[1,:,3], label='Score @ learning rate 1.5')
    ax.plot(results[2,:,1], results[2,:,3], label='Score @ learning rate 2.0')
    ax.set(xlim=(50, 250), ylim=(0, 1.1),
           ylabel='Accuracy Score', xlabel='Iterations',
           title='Boosting Accuracy Score Vs Num Iterations');
    plt.legend(loc='best')
    plt.savefig('02BoostingScoreVSIterations_'+dataset+'.png')
    # plt.show()
    plt.cla()
    ax1 = plt.axes()
    ax1.plot(results[0,:,1], results[0,:,4], label='Time @ learning rate 1.0')
    ax1.plot(results[1,:,1], results[1,:,4], label='Time @ learning rate 1.5')
    ax1.plot(results[2,:,1], results[2,:,4], label='Time @ learning rate 2.0')
    ax1.set(xlim=(50, 250), ylim=(0, 25),
           ylabel='Seconds', xlabel='Iterations',
           title='Boosting Time Taken Vs Num Iterations');
    plt.legend(loc='best')
    plt.savefig('02BoostingTimeVSIterations_'+dataset+'.png')
    # plt.show()
    plt.cla()
    print("Boosting Accuracy Score Vs Max Depth plotted and saved to file 03BoostingScoreVSNumNeighbours_.png")

    results = []
    for training_size in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-training_size, random_state=42)
        bdt = AdaBoostClassifier(
            DecisionTreeClassifier(max_depth=7),
            n_estimators=100,
            learning_rate=1.0)
        start = datetime.datetime.now()
        bdt = bdt.fit(X_train, y_train)
        finish = datetime.datetime.now()
        results.append([training_size, bdt.score(X_train, y_train),bdt.score(X_test, y_test),(finish-start).total_seconds()])
    results = np.asarray(results)
    ax = plt.axes()
    ax.plot(results[:,0], results[:,1], label='Training Set Score')
    ax.plot(results[:,0], results[:,2], label='Test Set Score')
    ax.plot(results[:,0], results[:,3], label='Training Time')
    ax.set(xlim=(0.1, 0.9), ylim=(0, 10),
           ylabel='Accuracy Score/Seconds', xlabel='Training Size',
           title='Boosting Learning Curve');
    plt.legend(loc='best')
    plt.savefig('02BoostingLearningCurve_'+dataset+'.png')
    plt.cla()
    print("Boosting Learning Curve saved to file TreeLearningCurve_",dataset,".png")
