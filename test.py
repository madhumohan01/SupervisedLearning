import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
import datetime
start = datetime.datetime.now()
iris = load_iris()
print(iris)
finish = datetime.datetime.now()
print((finish-start).total_seconds())
# print(iris.feature_names)
# print(iris.target_names)
# print(iris.class_names)
# x = np.linspace(0, 10, 1000)
# print(x)
# print(np.sin(x))
# ax = plt.axes()
# ax.plot(x, np.sin(x))
# ax.set(xlim=(0, 10), ylim=(-2, 2),
#        xlabel='x', ylabel='sin(x)',
#        title='A Simple Plot');
# plt.show()
# # plt.savefig('foo.png')
