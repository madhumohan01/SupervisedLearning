import numpy as np
import csv
from sklearn import preprocessing
import pandas as pd

def readDataFromFile(file_name):
    # with open(file_name, newline='') as csvfile:
    #     data = list(csv.reader(csvfile))
    data = np.genfromtxt(file_name, delimiter=';')
    # print(data.shape)
    # print(data)
    # print(np.any(np.isnan(data)))
    data = data[1:]
    # print(data)
    X = data[:,0:-1]
    y = data[:,-1]
    # print(X.shape)
    # print(y.shape)
    return X,y

def readDataFromCSV(file_name):
    with open(file_name, newline='') as csvfile:
        data = list(csv.reader(csvfile))
    data = np.asarray(data)
    # print(data)
    le = preprocessing.LabelEncoder()
    # for i in range(15):
    #     data[:,i] = le.fit_transform(data[:,i])
    print("Transforming categorical data with LabelEncoder...")
    data[:,1] = le.fit_transform(data[:,1])
    print("Workclass transformed as:", list(le.classes_))
    data[:,3] = le.fit_transform(data[:,3])
    print("Education transformed as:", list(le.classes_))
    data[:,5] = le.fit_transform(data[:,5])
    print("Marital Status transformed as:", list(le.classes_))
    data[:,6] = le.fit_transform(data[:,6])
    print("Occupation transformed as:", list(le.classes_))
    data[:,7] = le.fit_transform(data[:,7])
    print("Relationship transformed as:", list(le.classes_))
    data[:,8] = le.fit_transform(data[:,8])
    print("Race transformed as:", list(le.classes_))
    data[:,9] = le.fit_transform(data[:,9])
    print("Sex transformed as:", list(le.classes_))
    data[:,13] = le.fit_transform(data[:,13])
    print("Native Country transformed as:", list(le.classes_))
    data[:,14] = le.fit_transform(data[:,14])
    print("Target transformed as:", list(le.classes_))
    # print(data)
    print(data[0])
    print("Saving data to adult1.csv")
    pd.DataFrame(data).to_csv("adult1.csv", index=False)
    X = data[:,0:-1]
    y = data[:,-1]
    # print(X.shape)
    # print(y.shape)
    return X,y
