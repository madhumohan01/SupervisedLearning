# Supervised Learning
## Overview
Following 5 learning algorithms are implemented: 
* Decision trees with pruning
* Neural networks
* Boosting
* Support Vector Machines
* k-nearest neighbors
Datasets were Adult and Wine data sets from UCI machine learning repository. 
## Steps to Run
1. Install following requirements:
* numpy == 1.15.1
* scipy == 1.1.0
* scikit-learn == 0.20.0
* pandas == 0.23.4
* xlrd == 0.9.0
* matplotlib == 2.2.3
* seaborn == 0.9.0
* scikit-optimize == 0.5.2
* kneed == 0.1.0

2. Run following files with jython to create the data files
* DecisionTrees.py
* NeuralNetworks.py
* Boosting.py
* SVM.py
* KNN.py
## Results Obtained
![DecisionTree](/results/01DecisionTreeScoreVSDepth_ADULT.png)
![NeuralNetworks](/results/04AnnScoreVSNeurons_ADULT.png)
![Boosting](/results/02BoostingScoreVSIterations_ADULT.png)
![SVM](/results/05SVCRBFScoreGamma_ADULT.png)
![KNN](/results/03KnnScoreVSNumNeighbours_ADULT.png)
