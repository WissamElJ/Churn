
import pandas as pd 
import numpy as np 
from pandas import MultiIndex, Int64Index
from copy import deepcopy
import time
from datetime import date
from tabulate import tabulate


from sklearn.utils import class_weight
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier, GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import LinearSVC,SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score,f1_score, ConfusionMatrixDisplay,precision_score,recall_score,f1_score,classification_report,roc_curve,plot_roc_curve,auc,precision_recall_curve,plot_precision_recall_curve,average_precision_score
from sklearn.model_selection import cross_val_score , cross_val_predict, StratifiedKFold,train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import  CalibratedClassifierCV 
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from imblearn.ensemble import BalancedRandomForestClassifier,BalancedBaggingClassifier,RUSBoostClassifier

from imblearn.under_sampling import NearMiss,RandomUnderSampler
from imblearn.over_sampling import SMOTENC
import xgboost as xgb
from xgboost import XGBClassifier


import warnings
warnings.filterwarnings("ignore")



# For train
def modeling(models, X_train,y_train,X_test,y_test):   
    lst_1 = []
    
    for m in range(len(models)):
        lst_2 = []
        model = models[m][1]
        start = time.time()
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1]
        cm = confusion_matrix(y_test,y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        AUC_cv = cross_val_score(estimator= model, X = X_train,y = y_train, cv=5,scoring = 'roc_auc')#,fit_params=parameters)
        end = time.time()
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
        AUC = auc(false_positive_rate, true_positive_rate)
        
        print(models[m][0],':')
        print(cm)
        print('AUC: {:.3f}'.format(AUC))
        print('5-CV AUC: {:.3f}'.format(AUC_cv.mean())) 
        print('5-CV AUC Standard Deviation: {:.3f}'.format(AUC_cv.std())) 

                
        lst_2.append(models[m][0])
        lst_2.append(AUC)
        
        lst_2.append(AUC_cv.mean())
        lst_2.append(AUC_cv.std())
        lst_2.append(f1_score(y_test, y_pred))
        lst_2.append(accuracy_score(y_test, y_pred))
        
        lst_2.append(recall_score(y_test, y_pred))
        lst_2.append(precision_score(y_test, y_pred))
        

        lst_1.append(lst_2)
        df2 = pd.DataFrame(lst_1,columns=['Model','AUC','5-CV AUC','5-CV AUC std','F1','Accuracy','Recall','Precision'])

        df2.sort_values(by=['AUC'],inplace=True,ascending=False)
        df2.reset_index(drop = True).round(decimals = 3)
    return df2

#For adding class weights to fit
def modeling4(models, X_train,y_train,X_test,y_test,classes_weights):   
    lst_1 = []
    
    for m in range(len(models)):
        lst_2 = []
        model = models[m][1]
        model.fit(X_train,y_train,sample_weight=classes_weights)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1]
        cm = confusion_matrix(y_test,y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        AUC_cv = cross_val_score(estimator= model, X = X_train,y = y_train, cv=5,scoring = 'roc_auc')#,fit_params=parameters)

        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
        AUC = auc(false_positive_rate, true_positive_rate)
        
        print(models[m][0],':')
        print(cm)
        print('AUC: {:.3f}'.format(AUC))
        print('5-CV AUC: {:.3f}'.format(AUC_cv.mean())) 
        print('5-CV AUC Standard Deviation: {:.3f}'.format(AUC_cv.std())) 

                
        lst_2.append(models[m][0])
        lst_2.append(AUC)
        lst_2.append(AUC_cv.mean())
        lst_2.append(AUC_cv.std())
        lst_2.append(f1_score(y_test, y_pred))
        lst_2.append(accuracy_score(y_test, y_pred))
        
        lst_2.append(recall_score(y_test, y_pred))
        lst_2.append(precision_score(y_test, y_pred))
        

        lst_1.append(lst_2)
        df2 = pd.DataFrame(lst_1,columns=['Model','AUC','5-CV AUC','5-CV AUC std','F1','Accuracy','Recall','Precision'])

        df2.sort_values(by=['AUC'],inplace=True,ascending=False)
        df2.reset_index(drop = True).round(decimals = 3)
    return df2

#For oversampled training data
def modeling2(models, X_train,y_train,X_test,y_test):   
    lst_1 = []
    
    for m in range(len(models)):
        lst_2 = []
        model = models[m][1]
        model.fit(X_train_OS,y_train_OS)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1]
        cm = confusion_matrix(y_test,y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        AUC_cv = cross_val_score(estimator= model, X = X_train,y = y_train, cv=5,scoring = 'roc_auc')#,fit_params=parameters)

        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
        AUC = auc(false_positive_rate, true_positive_rate)
        
        print(models[m][0],':')
        print(cm)
        print('AUC: {:.3f}'.format(AUC))
        print('5-CV AUC: {:.3f}'.format(AUC_cv.mean())) 
        print('5-CV AUC Standard Deviation: {:.3f}'.format(AUC_cv.std())) 

                
        lst_2.append(models[m][0])
        lst_2.append(AUC)
        lst_2.append(AUC_cv.mean())
        lst_2.append(AUC_cv.std())
        lst_2.append(f1_score(y_test, y_pred))
        lst_2.append(accuracy_score(y_test, y_pred))
        
        lst_2.append(recall_score(y_test, y_pred))
        lst_2.append(precision_score(y_test, y_pred))
        

        lst_1.append(lst_2)
        df2 = pd.DataFrame(lst_1,columns=['Model','AUC','5-CV AUC','5-CV AUC std','F1','Accuracy','Recall','Precision'])

        df2.sort_values(by=['AUC'],inplace=True,ascending=False)
        df2.reset_index(drop = True).round(decimals = 3)
    return df2

# For undersampled training data
def modeling3(models, X_train,y_train,X_test,y_test):   
    lst_1 = []
    
    for m in range(len(models)):
        lst_2 = []
        model = models[m][1]
        model.fit(X_train_US,y_train_US)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1]
        cm = confusion_matrix(y_test,y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        AUC_cv = cross_val_score(estimator= model, X = X_train,y = y_train, cv=5,scoring = 'roc_auc')#,fit_params=parameters)

        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
        AUC = auc(false_positive_rate, true_positive_rate)
        
        print(models[m][0],':')
        print(cm)
        print('AUC: {:.3f}'.format(AUC))
        print('5-CV AUC: {:.3f}'.format(AUC_cv.mean())) 
        print('5-CV AUC Standard Deviation: {:.3f}'.format(AUC_cv.std())) 

                
        lst_2.append(models[m][0])
        lst_2.append(AUC)
        lst_2.append(AUC_cv.mean())
        lst_2.append(AUC_cv.std())
        lst_2.append(f1_score(y_test, y_pred))
        lst_2.append(accuracy_score(y_test, y_pred))
        
        lst_2.append(recall_score(y_test, y_pred))
        lst_2.append(precision_score(y_test, y_pred))
        

        lst_1.append(lst_2)
        df2 = pd.DataFrame(lst_1,columns=['Model','AUC','5-CV AUC','5-CV AUC std','F1','Accuracy','Recall','Precision'])

        df2.sort_values(by=['AUC'],inplace=True,ascending=False)
        df2.reset_index(drop = True).round(decimals = 3)
    return df2

