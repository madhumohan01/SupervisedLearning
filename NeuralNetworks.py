from sklearn.neural_network import MLPClassifier
from helpers import readDataFromCSV
from helpers import readDataFromFile
from sklearn.model_selection import train_test_split
import datetime
import numpy as np
import matplotlib.pyplot as plt

print("Starting processing ANN...")
for datasetid in range(2):
    if datasetid == 0:
        X, y = readDataFromCSV('adult.csv')
        dataset = 'ADULT'
        feature_names = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country']
    else:
        X, y = readDataFromFile('winequality-white.csv')
        feature_names = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']
        dataset = 'WINE'
    X = X.astype('float')
    y = y.astype('float')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    results1=[]
    results2=[]
    results3=[]
    for i in [10,20,30,50,75,100,150]:
    # for i in [10,20,30,40]:
        print("Processing Neurons", i, "Single Layer")
        clf = MLPClassifier(hidden_layer_sizes=(i,))
        start = datetime.datetime.now()
        clf.fit(X_train, y_train)
        finish = datetime.datetime.now()
        results1.append([i, clf.score(X_train, y_train),clf.score(X_test, y_test),(finish-start).total_seconds()])

        print("Processing Neurons", i, "2 Layers")
        clf = MLPClassifier(hidden_layer_sizes=(i,i))
        start = datetime.datetime.now()
        clf.fit(X_train, y_train)
        finish = datetime.datetime.now()
        results2.append([i, clf.score(X_train, y_train),clf.score(X_test, y_test),(finish-start).total_seconds()])

        print("Processing Neurons", i, "3 Layers")
        clf = MLPClassifier(hidden_layer_sizes=(i,i,i))
        start = datetime.datetime.now()
        clf.fit(X_train, y_train)
        finish = datetime.datetime.now()
        results3.append([i, clf.score(X_train, y_train),clf.score(X_test, y_test),(finish-start).total_seconds()])

    results1 = np.asarray(results1)
    results2 = np.asarray(results2)
    results3 = np.asarray(results3)
    ax = plt.axes()
    ax.plot(results1[:,0], results1[:,2], label='Test Score @ 1 Layer')
    ax.plot(results2[:,0], results2[:,2], label='Test Score @ 2 Layers')
    ax.plot(results3[:,0], results3[:,2], label='Test Score @ 3 Layers')
    ax.set(xlim=(10, 150), ylim=(0, 1.1),
           ylabel='Accuracy Score/Seconds', xlabel='Number of Neurons',
           title='ANN Accuracy Score/Time Taken Vs Number of Neurons');
    plt.legend(loc='best')
    plt.savefig('04AnnScoreVSNeurons_'+dataset+'.png')
    # plt.show()
    plt.cla()
    ax1 = plt.axes()
    ax1.plot(results1[:,0], results1[:,3], label='Training Time @ 1 Layer')
    ax1.plot(results2[:,0], results2[:,3], label='Training Time @ 2 Layers')
    ax1.plot(results3[:,0], results3[:,3], label='Training Time @ 3 Layers')
    ax1.set(xlim=(10, 150), ylim=(0, 30),
           ylabel='Seconds', xlabel='Iterations',
           title='ANN Time Taken Vs Num Iterations');
    plt.legend(loc='best')
    plt.savefig('04AnnTimeVSNeurons_'+dataset+'.png')
    plt.cla()
    print("ANN Time Taken Vs Max Depth plotted and saved to file 04AnnTimeVSNeurons.png")

    results = []
    for training_size in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
        print("Building Learning Curve for training_size", training_size)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-training_size, random_state=42)
        clf = MLPClassifier(hidden_layer_sizes=(20,20))
        start = datetime.datetime.now()
        clf.fit(X_train, y_train)
        finish = datetime.datetime.now()
        results.append([training_size, clf.score(X_train, y_train),clf.score(X_test, y_test),(finish-start).total_seconds()])
    results = np.asarray(results)
    ax = plt.axes()
    ax.plot(results[:,0], results[:,1], label='Training Set Score')
    ax.plot(results[:,0], results[:,2], label='Test Set Score')
    ax.plot(results[:,0], results[:,3], label='Training Time')
    ax.set(xlim=(0.1, 0.9), ylim=(0, 3),
           ylabel='Accuracy Score/Seconds', xlabel='Training Size',
           title='ANN Learning Curve');
    plt.legend(loc='best')
    plt.savefig('04ANNLearningCurve_'+dataset+'.png')
    plt.cla()
    print("ANN Learning Curve saved to file LearningCurve_",dataset,".png")
