import pandas as pd
import numpy as np

import seaborn as sns	
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt                                                                  
import matplotlib.colors as colors                                                               
																								 
from sklearn.utils import resample                                                               
from sklearn.model_selection import train_test_split                                             
from sklearn.preprocessing import scale                                                          
from sklearn.svm import SVC  #dont work
from sklearn.model_selection import GridSearchCV                                                 
from sklearn.decomposition import PCA                                                            
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression

# from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import RandomizedSearchCV 

import pickle

def evaluate(ytest, ypred):
	print("accuracy_score: ", accuracy_score(ytest, ypred))
	print('precision_score', precision_score(ytest, ypred))
	print('recall_score', recall_score(ytest, ypred))
	print('confusion matrix:\n', confusion_matrix(ytest, ypred))
	# ConfusionMatrixDisplay.from_estimator(svmtrain, Xtestscaled, ytest, values_format ='d', display_labels= ['flood', 'No Flood'])	# this is the evaluation of the model with test data(30rows)
	# plt.show()

def add_interactions(X):
	features = X.columns
	m = len(features)
	X_int = X.copy(deep=True)

	for i in range(m):
		
		feature_i_name = features[i]
		
		feature_i_data = X[feature_i_name]
		
		for j in range(i+1, m):
			
			feature_j_name = features[j]
			feature_j_data = X[feature_j_name]
			feature_i_j_name = feature_i_name+"_x_"+feature_j_name
			X_int[feature_i_j_name] =  feature_i_data * feature_j_data
		
	return X_int  # feature engineering. Adding more features!


data = pd.read_csv("./flood_data_new.csv")

# correl= data.corr()		# showing correlations betweent the columns
# plt.subplots(figsize= (15,15))
# sns.heatmap(correl, annot= True)
# plt.show()

#dataset development

#removing the date column

data.drop('Date', axis= 1, inplace= True)

X= data.drop("Floodornot", axis=1)
y= data["Floodornot"]

# split the data to test n train

Xtrain, Xtest, ytrain, ytest= train_test_split(X,y)


x_train_more = add_interactions(Xtrain)	# FEATURE engineering 
x_test_more  = add_interactions(Xtest)

Xtrainscaled= scale(x_train_more)		
Xtestscaled= scale(x_test_more)  
# print(Xtrainscaled, Xtestscaled)
#removing NaN type data:

while np.isnan(Xtrainscaled).sum():
	for i in range(len(Xtrainscaled)):
		if np.isnan(Xtrainscaled[i].sum()):
			Xtrainscaled= np.delete(Xtrainscaled,i,0)
			print("breaking")
			break
while np.isnan(Xtestscaled).sum():
	for i in range(len(Xtestscaled)):
		if np.isnan(Xtestscaled[i].sum()):
			Xtestscaled= np.delete(Xtestscaled,i,0)
			print("breaking")
			break

print("ytrain:\n", ytrain)
#building the baseline model on logistic regression

print("Baseline model with LogisticRegression: ")
logisreg= LogisticRegression(max_iter=1000)
logisreg.fit(Xtrainscaled, ytrain)

ypred= logisreg.predict(Xtestscaled)
evaluate(ytest, ypred)
# accuracy_score:  0.9444444444444444                                                                           
# precision_score 0.9090909090909091                                                                            
# recall_score 0.975609756097561                                                                                
# confusion matrix:                                                                                             
#  [[45  4]                                                                                                     
#  [ 1 40]]  


# using gridsearchCV
print("GridSearchCV : ")
gridparams= [ # trying random values
	{
		'C': [0.5, 1, 10, 100], # has to >0
		'gamma': ['scale', 1, 0.1, 0.01, 0.001, 0.0001],
		'kernel': ['rbf']},
]
# grid= GridSearchCV( SVC(), gridparams, cv=5, scoring='accuracy')
grid= GridSearchCV( SVC(), gridparams, refit=True )  #verbose=3
grid.fit(Xtrainscaled, ytrain)
print(grid.best_params_) 

gridsvmtrain= SVC(C=grid.best_params_['C'], gamma=grid.best_params_['gamma'])# training with the ideal paramters
gridsvmtrain.fit(Xtrainscaled, ytrain)
ypred= gridsvmtrain.predict(Xtestscaled)
evaluate(ytest, ypred)
# accuracy_score:  0.9555555555555556                                                                                
# precision_score 0.9302325581395349                                                                                 
# recall_score 0.975609756097561                                                                                     
# confusion matrix:                                                                                                  
#  [[46  3]                                                                                                          
#  [ 1 40]]  


#using random forest:

# print("RandomForestClassifier")
# def randomized_search(params, runs=20, clf=DecisionTreeClassifier(random_state=2)): 
#     rand_clf = RandomizedSearchCV(clf, params, n_iter=runs, cv=5, n_jobs=-1, random_state=2) 
#     rand_clf.fit(Xtrainscaled, ytrain) 
#     best_model = rand_clf.best_estimator_
#     best_score = rand_clf.best_score_

#     print("Training score: {:.3f}".format(best_score))
#     y_pred = best_model.predict(Xtestscaled)
#     accuracy = accuracy_score(ytest, y_pred)
#     print('Test score: {:.3f}'.format(accuracy))
    
#     return best_model


# randomized_search(params={
#                          'min_samples_leaf':[1,2,4,6,8,10,20,30],
#                           'min_impurity_decrease':[0.0, 0.01, 0.05, 0.10, 0.15, 0.2],
#                           'max_features':['auto', 0.8, 0.7, 0.6, 0.5, 0.4],
#                           'max_depth':[None,2,4,6,8,10,20], 
#                          }, clf=RandomForestClassifier(random_state=2))

# forrest = RandomForestClassifier(max_depth=2, max_features=0.5,
#                        min_impurity_decrease=0.01, min_samples_leaf=10,
#                        random_state=2) 
# forrest.fit(Xtrainscaled, ytrain)  
# pred5 = forrest.predict(Xtestscaled) 
# evaluate(ytest, pred5)



pickle.dump(grid, open('./model.pkl','wb'))

