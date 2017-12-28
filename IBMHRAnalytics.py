import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn import ensemble
import xgboost as xgb
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import roc_curve, auc, recall_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action="ignore")
np.random.seed(7)


# Creating training and testing dataset
dataset=pd.read_csv("/home/inderpreet/Pictures/HR-Employee-Attrition.csv")
dataset['Attrition'] = dataset['Attrition'].map({'Yes': 1, 'No': 0})
print(dataset.Attrition.value_counts())

#Dropping unnecessary columns (Single Value for each record)
dataset=dataset.drop(['EmployeeCount','EmployeeNumber', 'Over18', 'StandardHours'],axis=1)

# One-Hot Encoding
dataset_encoded=pd.get_dummies(dataset)

# Creating testing dataset
train_features,test_features,train_class,test_class=train_test_split(
                                                    dataset_encoded.drop('Attrition',1),
                                                    dataset_encoded['Attrition'],
                                                    test_size=0.2,
                                                    random_state=5
                                                    )

#Performing Feature Selection on Training Dataset
train_features_temp = train_features.copy(deep=True)  # Make a deep copy of the Training Data dataframe
selector = VarianceThreshold(0.09)
selector.fit(train_features_temp)
X_res = train_features_temp.loc[:, selector.get_support(indices=False)]
train_features=X_res
print("Number of Features in training dataset(after feature selection):-",train_features.shape[1])

#Selecting Same Features for testing dataset
test_features=test_features[train_features.columns]

# Creating training and cross-validation dataset
X_train,X_CV,Y_train,Y_CV=train_test_split(train_features,train_class)

# Applying Logistic Regression on imbalanced datasets
log_reg=LogisticRegression()
log_reg=log_reg.fit(X_train, Y_train)
log_pred=log_reg.predict(X_CV)
fpr, tpr, thresholds = roc_curve(Y_CV, log_pred)
roc_auc_log=auc(fpr,tpr)

# Upsampling training data using SMOTE
sm = SMOTE(random_state=12, ratio = 1.0)
x_res, y_res = sm.fit_sample(X_train, Y_train)
print (np.bincount(y_res))

# Applying Logistic Regression on Upsampled data
log_reg=LogisticRegression()
log_reg=log_reg.fit(x_res, y_res)
log_pred=log_reg.predict(X_CV)
fpr2, tpr2, thresholds = roc_curve(Y_CV, log_pred)
roc_auc_log_sm=auc(fpr2,tpr2)

# Area under ROC curves to show difference by using upsampling
plt.figure(num=None, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
plt.rcParams.update({'font.size': 14})
plt.xlabel('FP rate', fontsize=14, color='red')
plt.ylabel('TP rate', fontsize=14, color='red')
plt.ylim([0.0, 1.0])
plt.xlim([0.0, 1.0])
plt.title('ROC curve', fontsize=14, color='red')
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k')
plt.plot(fpr, tpr, label='Logistic Regression: ' + str(roc_auc_log)[0:7])
plt.plot(fpr2, tpr2, label='Logistic Regression (SMOTE): ' + str(roc_auc_log_sm)[0:7])
plt.legend(fontsize=14)
plt.legend(fontsize=14)
plt.grid(True)
plt.show()


# Now on using Upsampled data
# Setting Paramters corresponding to the plot
plt.figure(num=None, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
plt.rcParams.update({'font.size': 14})
plt.xlabel('FP rate', fontsize=14, color='red')
plt.ylabel('TP rate', fontsize=14, color='red')
plt.ylim([0.0, 1.0])
plt.xlim([0.0, 1.0])
plt.title('ROC curve', fontsize=14, color='red')
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k')

# Applying Various Models


# Applying Logistic Regression
log_reg=LogisticRegression()
log_reg=log_reg.fit(x_res, y_res)
log_pred=log_reg.predict(X_CV)
fpr_log, tpr_log, thresholds = roc_curve(Y_CV, log_pred)
roc_auc_log=auc(fpr_log,tpr_log)
plt.plot(fpr_log, tpr_log, label='Logistic Regression: ' + str(roc_auc_log)[0:7])


# Applying Decision Tree
decision_reg = DecisionTreeClassifier(criterion='gini',
                                            max_depth=10, max_leaf_nodes=23, splitter='best')
decision_reg=decision_reg.fit(x_res, y_res)
decision_pred=decision_reg.predict(X_CV)
fpr_dec, tpr_dec, thresholds = roc_curve(Y_CV, decision_pred)
roc_auc_dec=auc(fpr_log,tpr_log)
plt.plot(fpr_dec, tpr_dec, label='Decision Tree: ' + str(roc_auc_dec)[0:7])


# Applying Gaussian Naive Bayes
naivebay_reg = GaussianNB()
naivebay_reg=naivebay_reg.fit(x_res, y_res)
naivebay_pred=naivebay_reg.predict(X_CV)
fpr_nb, tpr_nb, thresholds = roc_curve(Y_CV, naivebay_pred)
roc_auc_nb=auc(fpr_nb,tpr_nb)
plt.plot(fpr_nb, tpr_nb, label='Gaussian Naive Bayes: ' + str(roc_auc_nb)[0:7])


# Applying Support Vector Machines
svm_reg = SVC()
svm_reg=svm_reg.fit(x_res, y_res)
svm_pred=svm_reg.predict(X_CV)
fpr_svm, tpr_svm, thresholds = roc_curve(Y_CV, svm_pred)
roc_auc_svm=auc(fpr_svm,tpr_svm)
plt.plot(fpr_svm, tpr_svm, label='Support Vector Machines: ' + str(roc_auc_svm)[0:7])


# Applying AdaBoost Classifier
ada_reg = AdaBoostClassifier(random_state=2, n_estimators=500)
ada_reg=ada_reg.fit(x_res, y_res)
ada_pred=ada_reg.predict(X_CV)
fpr_ada, tpr_ada, thresholds = roc_curve(Y_CV, ada_pred)
roc_auc_ada=auc(fpr_ada,tpr_ada)
plt.plot(fpr_ada, tpr_ada, label='Adaboost Classifier: ' + str(roc_auc_ada)[0:7])


# Applying Gradient Boosting Classifier
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2, 'learning_rate': 0.01}
gb_reg = ensemble.GradientBoostingClassifier(**params)
gb_reg=gb_reg.fit(x_res, y_res)
gb_pred=gb_reg.predict(X_CV)
fpr_gb, tpr_gb, thresholds = roc_curve(Y_CV, gb_pred)
roc_auc_gb=auc(fpr_gb,tpr_gb)
plt.plot(fpr_ada, tpr_ada, label='Gradient Boosting Classifier: ' + str(roc_auc_gb)[0:7])


# Applying Random Forest Classifier
rf_reg = RandomForestClassifier(random_state=1)
rf_reg=rf_reg.fit(x_res, y_res)
rf_pred=rf_reg.predict(X_CV)
fpr_rf, tpr_rf, thresholds = roc_curve(Y_CV, rf_pred)
roc_auc_rf=auc(fpr_rf,tpr_rf)
plt.plot(fpr_ada, tpr_ada, label='Random Forest Classifier: ' + str(roc_auc_rf)[0:7])


# # Applying XGBoost Classifier
# X_CV=X_CV[X_train.columns]
# xgb_reg = xgb.XGBClassifier()
# xgb_reg=xgb_reg.fit(x_res, y_res)
# xgb_pred=xgb_reg.predict(X_CV)
# fpr_xgb, tpr_xgb, thresholds = roc_curve(Y_CV, xgb_pred)
# roc_auc_xgb=auc(fpr_xgb,tpr_xgb)
# plt.plot(fpr_xgb, tpr_xgb, label='XGBoost Classifier: ' + str(roc_auc_xgb)[0:7])


#Applying Majority Voting Concept
majority_class = VotingClassifier(estimators=[('lr', log_reg),
                                    ('gnb', gb_reg),('naive_bayes',naivebay_reg)],
                                    voting='hard')
majority_class = majority_class.fit(x_res, y_res)
majority_pred=majority_class.predict(X_CV)
fpr_majority, tpr_majority, thresholds = roc_curve(Y_CV, majority_pred)
roc_auc_majority=auc(fpr_majority,tpr_majority)
plt.plot(fpr_majority, tpr_majority, label='Majority Voting Classifier (Soft): ' + str(roc_auc_majority)[0:7])


plt.legend(fontsize=14)
plt.legend(fontsize=14)
plt.grid(True)
plt.show()


#Applying Best Model on Testing Dataset
majority_class = majority_class.fit(train_features, train_class)
majority_pred=majority_class.predict(test_features)
fpr_majority, tpr_majority, thresholds = roc_curve(test_class, majority_pred)
roc_auc_majority=auc(fpr_majority,tpr_majority)
plt.plot(fpr_majority, tpr_majority, label='Majority Voting Classifier (Test Dataset): ' + str(roc_auc_majority)[0:7])

plt.legend(fontsize=14)
plt.legend(fontsize=14)
plt.grid(True)
plt.show()
