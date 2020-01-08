import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
# import csv
import graphviz
import matplotlib.pyplot as plt
import datetime
from helpers import readDataFromFile
from helpers import readDataFromCSV
import random

# Source: https://datascience.stackexchange.com/questions/19842/anyway-to-know-all-details-of-trees-grown-using-randomforestclassifier-in-scikit/36228#36228
def dectree_max_depth(tree):
    n_nodes = tree.node_count
    children_left = tree.children_left
    children_right = tree.children_right

    def walk(node_id):
        if (children_left[node_id] != children_right[node_id]):
            left_max = 1 + walk(children_left[node_id])
            right_max = 1 + walk(children_right[node_id])
            return max(left_max, right_max)
        else: # leaf
            return 1

    root_node_id = 0
    return walk(root_node_id)


# X, y = readDataFromFile('adult.data')
for datasetid in range(2):
    if datasetid == 0:
        X, y = readDataFromCSV('adult.csv')
        dataset = 'ADULT'
        feature_names = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country']
    else:
        X, y = readDataFromFile('winequality-white.csv')
        feature_names = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']
        dataset = 'WINE'
    print("Starting processing Decision Tree...")
    # class_names=['Less than 50K','More than 50K']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    print("Calculating Accuracy Score for different max_depth. This will take a couple of mins...")
    results = []
    for i in range(1,50):
        print("Calculating Accuracy Score for max_depth",i)
        clf = tree.DecisionTreeClassifier(max_depth=i)
        start = datetime.datetime.now()
        clf = clf.fit(X_train, y_train)
        finish = datetime.datetime.now()
        # print(i, clf.score(X_train, y_train),clf.score(X_test, y_test),dectree_max_depth(clf.tree_),clf.tree_.node_count)
        results.append([i, clf.score(X_train, y_train),clf.score(X_test, y_test),(finish-start).total_seconds()])
    results = np.asarray(results)
    ax = plt.axes()
    ax.plot(results[:,0], results[:,1], label='Training Set Score')
    ax.plot(results[:,0], results[:,2], label='Test Set Score')
    ax.plot(results[:,0], results[:,3], label='Training Time')
    ax.set(xlim=(1, 50), ylim=(0, 1.1),
           ylabel='Accuracy Score/Seconds', xlabel='Depth',
           title='Decision Tree Accuracy Score Vs Max Depth');
    plt.legend(loc='best')
    plt.savefig('01DecisionTreeScoreVSDepth_'+dataset+'.png')
    plt.cla()
    print("Accuracy Score Vs Max Depth plotted and saved to file 01DecisionTreeScoreVSDepth_",dataset,".png")

    print("Calculating accuracy scores...")
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    train_score_no_prune = clf.score(X_train, y_train)
    test_score_no_prune = clf.score(X_test, y_test)

    clf = tree.DecisionTreeClassifier(max_depth=7)
    clf = clf.fit(X_train, y_train)
    train_score_prune = clf.score(X_train, y_train)
    test_score_prune = clf.score(X_test, y_test)

    print("Accuracy Score Without Pruning:")
    print("For Training Set:",train_score_no_prune)
    print("For Testing Set:",test_score_no_prune)

    print("Accuracy Score with Pruning at Max_Depth 7:")
    print("For Training Set:",train_score_prune)
    print("For Testing Set:",test_score_prune)

    print("Building Decision Tree Graph upto depth 3...")
    clf = tree.DecisionTreeClassifier(max_depth=3)
    clf = clf.fit(X_train, y_train)
    # dot_data = tree.export_graphviz(clf, out_file=None)
    dot_data = tree.export_graphviz(clf, out_file=None,
        feature_names=feature_names,
        # class_names=class_names,
        filled=True, rounded=True,
        special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.format = 'png'
    graph.render("01DecisionTreeGraph_"+dataset)
    print("Building Decision Tree Graph saved as 01DecisionTreeGraph_",dataset,".png")

    print("Decision Tree Building Learning Curves")
    results1 = []
    results2 = []
    for training_size in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
        print("Building Learning Curve for training_size", training_size)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-training_size, random_state=42)
        clf = tree.DecisionTreeClassifier()
        start = datetime.datetime.now()
        clf = clf.fit(X_train, y_train)
        finish = datetime.datetime.now()
        results1.append([training_size, clf.score(X_train, y_train),clf.score(X_test, y_test),(finish-start).total_seconds()])

        clf = tree.DecisionTreeClassifier(max_depth=7)
        start = datetime.datetime.now()
        clf = clf.fit(X_train, y_train)
        finish = datetime.datetime.now()
        results2.append([training_size, clf.score(X_train, y_train),clf.score(X_test, y_test),(finish-start).total_seconds()])
    results1 = np.asarray(results1)
    results2 = np.asarray(results2)
    print(results1)
    ax = plt.axes()
    ax.plot(results1[:,0], results1[:,1], label='Training Set Score')
    ax.plot(results1[:,0], results1[:,2], label='Test Set Score')
    ax.plot(results1[:,0], results1[:,3], label='Training Time')
    ax.set(xlim=(0.1, 0.9), ylim=(0, 1.1),
           ylabel='Accuracy Score/Seconds', xlabel='Training Time',
           title='Decision Tree (No Pruning) Learning Curve');
    plt.legend(loc='best')
    plt.savefig('01DecisionTreeLearningCurveNoPruning_'+dataset+'.png')
    plt.cla()
    print("Learning Curve saved to file 01DecisionTreeLearningCurve_",dataset,".png")
    ax = plt.axes()
    ax.plot(results2[:,0], results2[:,1], label='Training Set Score')
    ax.plot(results2[:,0], results2[:,2], label='Test Set Score')
    ax.plot(results2[:,0], results2[:,3], label='Training Time')
    ax.set(xlim=(0.1, 0.9), ylim=(0, 1.1),
           ylabel='Accuracy Score/Seconds', xlabel='Training Size',
           title='Decision Tree (With Pruning) Learning Curve');
    plt.legend(loc='best')
    plt.savefig('01DecisionTreeLearningCurvePruning_'+dataset+'.png')
    plt.cla()
    print("Learning Curve saved to file 01DecisionTreeLearningCurve_",dataset,".png")
