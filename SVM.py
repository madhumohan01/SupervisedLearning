from sklearn.svm import SVC
from helpers import readDataFromFile
from helpers import readDataFromCSV
from sklearn.model_selection import train_test_split
import datetime
import numpy as np
import matplotlib.pyplot as plt

# X, y = readDataFromFile('winequality-white.csv')
print("Starting processing SVC...")
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
    results1 = []
    results2 = []
    results3 = []
    # for i in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
    print("Building Learning Curve...")
    # for i in [0.1,0.2,0.3]:
    for i in [0.3,0.7,1.0]:
        print("Building for Gamma",i)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # print("Building rbf")
        clf = SVC(gamma=i)
        start = datetime.datetime.now()
        clf.fit(X_train, y_train)
        finish = datetime.datetime.now()
        results1.append([i,clf.score(X_train, y_train),clf.score(X_test, y_test),(finish-start).total_seconds()])

    results1 = np.asarray(results1)
    # print results1
    ax = plt.axes()
    ax.plot(results1[:,0], results1[:,1], label='Training Set Score')
    ax.plot(results1[:,0], results1[:,2], label='Test Set Score')
    ax.plot(results1[:,0], results1[:,3], label='Training Time')
    ax.set(xlim=(0.3, 1.0), ylim=(0, 1.1),
           ylabel='Accuracy Score', xlabel='Gamma',
           title='SVC - rbf Accuracy Score Vs Gamma');
    plt.legend(loc='best')
    plt.savefig('05SVCRBFScoreGamma_'+dataset+'.png')
    plt.cla()
    print("SVC RBF Accuracy Score Vs Max Depth plotted and saved to file 05SVCRBFLearningCurve.png")

    ax = plt.axes()
    ax.plot(results1[:,0], results1[:,3], label='Training Time')
    ax.set(xlim=(0.3, 1.0), ylim=(0, 30),
           ylabel='Seconds', xlabel='Gamma',
           title='SVC - rbf Time Taken Vs Gamma');
    plt.legend(loc='best')
    plt.savefig('05SVCRBFTimeGamma_'+dataset+'.png')
    plt.cla()
    print("SVC RBF Accuracy Score Vs Max Depth plotted and saved to file 05SVCRBFLearningCurve.png")

    # results = []
    # for training_size in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
    #     print("Building Learning Curve for training_size", training_size)
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-training_size, random_state=42)
    #     clf = SVC(gamma='auto')
    #     start = datetime.datetime.now()
    #     clf.fit(X_train, y_train)
    #     finish = datetime.datetime.now()
    #     results.append([training_size, clf.score(X_train, y_train),clf.score(X_test, y_test),(finish-start).total_seconds()])
    # results = np.asarray(results)
    # ax = plt.axes()
    # ax.plot(results[:,0], results[:,1], label='Training Set Score')
    # ax.plot(results[:,0], results[:,2], label='Test Set Score')
    # # ax.plot(results[:,0], results[:,3], label='Training Time')
    # ax.set(xlim=(0.1, 0.9), ylim=(0, 1.1),
    #        ylabel='Accuracy Score', xlabel='Training Size',
    #        title='SVM Learning Curve');
    # plt.legend(loc='best')
    # plt.savefig('05SVMLearningCurve_'+dataset+'.png')
    # plt.cla()
    # print("SVM Learning Curve saved to file LearningCurve_",dataset,".png")
