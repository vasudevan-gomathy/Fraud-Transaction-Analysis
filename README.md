# Fraud-Transaction-Analysis
## Summary
This project involves analysing Transaction for predicting whether it is an fraudulent transaction or not by finding the patterns using Machine Learning Algorithms.

## Problem Statement
It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase.

## About the Dataset
The dataset contains transactions made by credit cards in September 2013 by European cardholders.
This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.

It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.

Given the class imbalance ratio, we recommend measuring the accuracy using the Area Under the Precision-Recall Curve (AUPRC). Confusion matrix accuracy is not meaningful for unbalanced classification.

Analysing the Dataset

The following shows the proportion of Fraudulent and not fraudulent Transaction:

![image](https://user-images.githubusercontent.com/85822284/203844372-428c9718-051f-42e2-9c6e-4444faba50c5.png)

SMOTE-ENN:

Since there is an imbalance in dataset. SMOTE has been used for balancing. Developed by Wilson (1972), the ENN method works by finding the K-nearest neighbor of each observation first, then check whether the majority class from the observation’s k-nearest neighbor is the same as the observation’s class or not. If the majority class of the observation’s K-nearest neighbor and the observation’s class is different, then the observation and its K-nearest neighbor are deleted from the dataset. In default, the number of nearest-neighbor used in ENN is K=3.

Machine Learning Models:

Different machine learning models have been used to find out the best model out of it. The following are the ML models used:

 - Logistic Regression
 - Decision Tree
 - Random Forest
 - Naive Bayes
 - Gradient Boosting 

Comparison of AUC-ROC score for all of these models:

![image](https://user-images.githubusercontent.com/85822284/203844760-526b64d9-128b-4fd4-90b5-0dc082bffdb9.png)


Out of these, AUC-ROC(Area Under The Curve - Receiver Operating Characteristics) has been taken to finalise Gradient Boosting Classifier Algorithm as the best model which has a score of about 93%. 

## Transaction Analysis App

This is an user friendly and easily customizable app that analyze whether the transaction is fraudulent or not. This app is created with the help of Streamlit.

### Screenshot of the App

![image](https://user-images.githubusercontent.com/85822284/203845385-e901e4f6-b8e0-468d-86d0-aeffb81b0213.png)


![image](https://user-images.githubusercontent.com/85822284/203845426-d418f68c-ff7a-420b-9b46-4d35f49dd93d.png)


This app feeds the transaction details uploaded by us to the Gradient Boosting Model and adds the analyzed report into new field which can be download for further inference. In addition to that, it provides visualization about the prediction in a pie chart.

## Conclusion:

Although the model has good AUC-ROC score of 93%, down the line it can be improved by:

 - Adding more data to the pipeline to further train the model.
