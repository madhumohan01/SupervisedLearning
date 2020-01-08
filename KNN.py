from sklearn.neighbors import KNeighborsClassifier
from helpers import readDataFromCSV
from helpers import readDataFromFile
from sklearn.model_selection import train_test_split
import datetime
import numpy as np
import matplotlib.pyplot as plt

print("Starting processing KNN...")
for datasetid in range(2):
    if datasetid == 0:
        X, y = readDataFromCSV('adult.csv')
        dataset = 'ADULT'
        feature_names = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country']
    else:
        X, y = readDataFromFile('winequality-white.csv')
        feature_names = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']
        dataset = 'WINE'
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    results = []
    print("Calculating Accuracy Score for different max_depth. This will take a couple of mins...")
    for i in range(1,50):
        print("Processing n_neighbours", i)
        neigh = KNeighborsClassifier(n_neighbors=i)
        start = datetime.datetime.now()
        neigh.fit(X_train, y_train)
        finish = datetime.datetime.now()
        # print(i, neigh.score(X_train, y_train),neigh.score(X_test, y_test))
        results.append([i, neigh.score(X_train, y_train),neigh.score(X_test, y_test),(finish-start).total_seconds()])

    results = np.asarray(results)
    ax = plt.axes()
    ax.plot(results[:,0], results[:,1], label='Training Set Score')
    ax.plot(results[:,0], results[:,2], label='Test Set Score')
    ax.plot(results[:,0], results[:,3], label='Training Time')
    ax.set(xlim=(1, 50), ylim=(0, 1.1),
           ylabel='Accuracy Score/Seconds', xlabel='Number of Neighbours',
           title='KNN Accuracy Score/Time Taken Vs Number of Neighbours');
    plt.legend(loc='best')
    plt.savefig('03KnnScoreVSNumNeighbours_'+dataset+'.png')
    plt.cla()
    print("KNN Accuracy Score Vs Max Depth plotted and saved to file 01KnnVSNumNeighbours.png")

    print("Calculating accuracy scores...")
    neigh = KNeighborsClassifier(n_neighbors=12)
    neigh.fit(X_train, y_train)
    print("Accuracy Score with number of neigbours 12:")
    print("For Training Set:",neigh.score(X_train, y_train))
    print("For Testing Set:",neigh.score(X_test, y_test))

    results = []
    for training_size in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
        print("Building Learning Curve for training_size", training_size)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-training_size, random_state=42)
        neigh = KNeighborsClassifier(n_neighbors=12)
        start = datetime.datetime.now()
        neigh.fit(X_train, y_train)
        finish = datetime.datetime.now()
        results.append([training_size, neigh.score(X_train, y_train),neigh.score(X_test, y_test),(finish-start).total_seconds()])
    results = np.asarray(results)
    ax = plt.axes()
    ax.plot(results[:,0], results[:,1], label='Training Set Score')
    ax.plot(results[:,0], results[:,2], label='Test Set Score')
    ax.plot(results[:,0], results[:,3], label='Training Time')
    ax.set(xlim=(0.1, 0.9), ylim=(0, 1.1),
           ylabel='Accuracy Score/Seconds', xlabel='Training Size',
           title='KNN Learning Curve');
    plt.legend(loc='best')
    plt.savefig('03KNNLearningCurve_'+dataset+'.png')
    plt.cla()
    print("KNN Learning Curve saved to file LearningCurve_",dataset,".png")
