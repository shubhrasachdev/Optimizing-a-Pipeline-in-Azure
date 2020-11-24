# Optimizing-an-ML-Pipeline-in-Azure

## Overview

This project is part of the Udacity Azure ML Nanodegree. In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model. This model is then compared to an Azure AutoML run.

## Summary
### Problem Statement

The given dataset is related with direct marketing campaigns bank. The aim is to predict whether a client will subscribe to a term deposit, denoted by the feature 'y', where 'y' would be subscribed ('yes') or not subscribed ('no').

### Solution

We attempt to solve this problem using Microsoft Azure's Hyperdrive and AutoML, and then compare the performance metrics presented by both. According to our results,the best performing model turned out to be an AutoML based Voting Ensemble model with an accuracy of 91.69% accuracy, versus a Hyperdrive based Logistic Regression model that presented an accuracy metric of 91.55%.

## Scikit-learn Pipeline

The hyperdrive run involves a series of steps to be completed in order to reach a meaningful result. They are:-
1. Importing the dataset - The desired dataset is located at https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv. We create a TabularDataset using TabularDatasetFactory, by utilizing its from_delimited_files() method.
2. Preprocessing - We clean the dataset for use (encoding the features, etc)
3. Splitting the dataset - We split the dataset into train and test sets, where the test size is 0.2.
4. We setup hyperparameter tuning for two parameters - inverse regularization strength 'C' and the maximum number of iterations taken for the solvers to converge 'max_iter' for a Logistic Regression Classifier. The values for C and max_iter were randomly selected from the search spaces [0.01, 0.1, 1, 10, 100, 1000] and [25, 50, 100, 150, 250] respectively.
5. Next, we define a Bandit early termination policy with a slack factor of 0.1 and evaluation_interval of 2.
6. The primary metric to be maximized for this Hyperdrive run is chosen to be 'Accuracy'.
7. The hyperdrive configured experiment is submitted for a run.

As a result of this run, we achieve an accuracy of 91.55% for C = 1 and max_iter = 50.

**What are the benefits of the chosen parameter sampler?**

Random Sampler promotes and supports early termination policies over a range, and allows picking a random value from a given range. This allows for tuning a hyperparameter more efficiently, as the sampler does not go through ever single value in the range. Hence, it is possible to train an optimal model in a shorter period of time.

**What are the benefits of the early stopping policy chosen?**

The Bandit Policy helps in termination of the hyperparameter tuning process, if there occurs a considerable drop in the performance of the model in terms of its chosen primary metric. This helps in eliminating the models that have sub-par performance.

## AutoML

Our AutoML experiment and its run trained 48 models. The child-run that gave the best performing model was a Voting Ensemble model with an accuracy of 91.69%. Voting ensemble uses numerous models to make predictions with weighted averages, and then picks the one that has the majority votes.
Here is a XGBoost Classifier part of the voting ensemble, along with its hyperparameters - 
```
37 - xgboostclassifier
{'base_score': 0.5,
 'booster': 'gbtree',
 'colsample_bylevel': 1,
 'colsample_bynode': 1,
 'colsample_bytree': 0.6,
 'eta': 0.2,
 'gamma': 0,
 'learning_rate': 0.1,
 'max_delta_step': 0,
 'max_depth': 2,
 'max_leaves': 0,
 'min_child_weight': 1,
 'missing': nan,
 'n_estimators': 100,
 'n_jobs': 1,
 'nthread': None,
 'objective': 'reg:logistic',
 'random_state': 0,
 'reg_alpha': 0,
 'reg_lambda': 1.0416666666666667,
 'scale_pos_weight': 1,
 'seed': None,
 'silent': None,
 'subsample': 0.9,
 'tree_method': 'auto',
 'verbose': -10,
 'verbosity': 0}
 ```
 
