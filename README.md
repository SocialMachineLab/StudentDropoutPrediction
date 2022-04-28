# Dropout Prediction

This project is for predicting student dropout in 3 degree (ADM, ARQ, and CSI) at a Brazilian university. By using Genetic Algorithm (GA)+ Support Vector machine (SVM) for feature selection and Long short-term memory (LSTM) for Time Series (TS) data training and test.

### How to cite this work?

Waiting for the upcoming reference.

### Imported Libraries
The scripts of projects are all based in Python 3.8 and anaconda package manager. The external dependencies are as following:

* torch
* torch.nn
* torch.utils.data
* torch.autograd
* torch.utils.data
* matplotlib.pyplot 
* sklearn.svm
* sklearn.model_selection
* sklearn.preprocessing
* sklearn.impute
* sklearn.ensemble

***Note: Install all libraries by using `conda install` or `pip install`***

### Pre-processing
`preprocessing.py` implements the pre-processing. `subset_seperator(self)` is separating raw dataset by degree; `organise(self)` is for data cleaning, data normalization and Data validation. `impute_missing_value(self)` is for data imputation.



### Feature Selection

`feature_selector.py` implements the feature selection by using GA+SVM. `get_fitness(self,pop,path)` is for getting fitness of each individual by SVM; `select(self,pop, fitness), crossover(self,parent, pop), mutate(self,child), and evolution(self)` are the steps of GA.


### Training+Test
`xx_dataloader.py, xx=ADM,ARQ, or CSI` implements the training and test by LSTM.

### Running project

***Directly Run `xx_dataloader.py, xx=ADM,ARQ, or CSI`*** to get the result of preprocessing, feature selection and training and test.


