# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 14:02:29 2022

@author: Chaoran
"""
import pandas as pd
import numpy as np
import maplotlib.pyplot as plt
from sklearn import *
from sklearn.linear_model import LogisticRegression

cr_loan=pd.read_csv(r'/Users//Documents/DBC/Credit_Data.csv')

# Explore the Credit Data

print(cr_loan.dtypes)
print(cr_loan.head(5))
print(pd.crosstab(cr_loan.loan_intent,cr_loan.loan_status,margins=True))


# Finding outliers with cross tables

print(pd.crosstab(cr_loan.loan_status,cr_loan.person_home_ownership,value=cr_loan.person_emp_length,aggfunc=np.max))
cr_loan.pivot_table(index='loan_status',columns='person_home_ownership',aggfunc=np.sum)
pd.pivot_table(cr_loan,values='age',columns='home_ownership',index='loan_status',aggfunc=np.count_nonzero)
plt.hist(cr_loan.int_rate.dropna(),bins=20)

plt.scatter(cr_loan['person_age'],cr_loan['loan_amnt'],c='blue',alpha=0.5)
plt.xlabel('Age')
plt.ylabel('Loan Amount')
plt.show()
 

# Replacing missing credit data

print(cr_loan.columns[cr_loan.isnull().any()])
print(cr_loan[cr_loan['person_emp_length'].isnull()])

null_columns=(cr_loan.columns[cr_loan.isnull().any()])
cr_loan[null_columns].isnull().sum()

cr_loan.person_emp_length.fillna(cr_loan.person_emp_length.median(),inplace=True)

indices = cr_loan[cr_loan['loan_int_rate'].isnull()].index
cr_loan.drop(indices,inplace=True)

indices=cr_loan[cr_loan['loan_status'].isnull()].index
cr_loan.drop(indices,inplace=True)

indices=cr_loan[cr_loan['person_age']>70].index
cr_loan.drop(indices,inplace=True)
cr_loan_clean=cr_loan.copy()


#Build Logistic Model

#Create the X and y data sets
X = cr_loan_clean[['loan_int_rate']]
y = cr_loan_clean[['loan_status']]

#Create and fit a logistic regression model
clf_logistic_single=LogisticRegression()
clf_logistic_single.fix(X,np.ravel(y))

#Print the parameters of the model
print(clf_logistic_single.get_params())

#print clf_logistic_single intercept of the model
print(clf_logistic_single.intercept_)


#Build Logistic Model

#Create X data for the model
X_multi = cr_loan[['loan_int_rate','person_emp_length']]

#Create a set of y for training
y = cr_loan[['loan_status']]

#Create and train a new logistic regression
clf_logistic_multi = LogisticRegression(solver='lbfgs').fit(X_multi, np.ravel(y))

#print the intercept of the model
print(clf_logistic_multi.intercept_)


# Creating training and test sets

#Create the X and y data sets
X = cr_loan_clean[['loan_int_rate','person_emp_length','person_income']]
y = cr_loan_clean[['loan_status']]

# Use test_train_split to create the training and test sets
X_train, X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size=.4,randomstate=1)

# Create and fit the logistic regression model
clf_logistic = LogisticRegression(solver='lbfgs').fit(X_train, np.ravel(y_train))

# print the models coefficients
print(clf_logistic.coef_)


#Build Logistic Model

X=cr_loan_clean.drop('loan_status',axis=1)
y=cr_loan_clean[['loan_status']]
X_train, X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size=.4,randomstate=1)

clf_logistic = LogisticRegression()
clf_logistic.fit(X_train[['loan_int_rate']], np.ravel(y_train))

clf_logistic_single.get_params()
clf_logistic_multi.intercept_
clf_logistic.coef_


# one-hot encoding credit data

cred_num = cr_loan_clean.select_dtypes(exclude=['object'])
cred_cat = cr_loan_clean.select_dtypes(include=['object'])
cred_cat_onehot = pd.get_dummies(cred_cat)
cr_loan_clean = pd.concat([cred_num,cred_cat_onehot],axis=1)


#Build logistic Model

X=cr_loan_clean.drop('loan_status',axis=1)
y=cr_loan_clean[['loan_status']]
X_train, X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size=.4,randomstate=1)

clf_logistic = LogisticRegression()
clf_logistic.fit(X_train[['loan_int_rate']], np.ravel(y_train))

clf_logistic_single.get_params()
clf_logistic_multi.intercept_
clf_logistic.coef_


#Credit model performance

preds=clf_logistic.predict_proba(X_test)
preds_df = pd.DataFrame(preds[:,0],columns=['prob_default'])
preds_df['loan_status'] = preds_df['prob_default'].apply(lambda x: 1 if x > 0.4 else 0)
print(metrics.classification_report(y_test,preds_df['loan_status']))

fallout,sensitivity,threshold = metrics.roc_curvery(y_test,preds[:,1])
auc=metrics.auc(y_test,preds[:,0])
plt.plot(fallout,sensitivity,color='darkorange')
plt.plot([0,1],[0,1],linestyle='--')
plt.show()

clf_logistic.score(X_test,y_test)