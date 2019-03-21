import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import Lasso

#read data 
data = pd.read_csv("/Users/snehamitta/Desktop/CharityML/census.csv")

#split into target and features
target = data['income']
features = data.drop('income', axis = 1)

skewed = ['capital-gain', 'capital-loss']
features_log_transformed = pd.DataFrame(data = features)
features_log_transformed[skewed] = features[skewed].apply(lambda x: np.log(x + 1))

scaler = MinMaxScaler() # default=(0, 1)
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

features_log_minmax_transform = pd.DataFrame(data = features_log_transformed)
features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])

features_final = pd.get_dummies(features_log_minmax_transform)

income = target.map({'<=50K': 0, '>50K':1})

print(features_final.head(5))


X_train, X_test, y_train, y_test = train_test_split(features_final, 
                                                    income, 
                                                    test_size = 0.2, 
                                                    random_state = 21)

print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))

clf = AdaBoostClassifier(random_state=42)

# Create the parameters list you wish to tune, using a dictionary if needed.
parameters = {'n_estimators': [200, 300, 500, 600]}

# Make an roc_auc scoring object using make_scorer()
scorer = make_scorer(roc_auc_score)

# Perform grid search on the classifier using 'scorer' as the scoring method using GridSearchCV()
grid_obj = GridSearchCV(clf, parameters, scoring=scorer, cv=5)

# Fit the grid search object to the training data and find the optimal parameters using fit()
grid_fit = grid_obj.fit(X_train, y_train)

# Get the estimator
best_clf = grid_fit.best_estimator_

print("best_estimator", grid_fit.best_estimator_)

# Make predictions using the unoptimized and model
predictions = (clf.fit(X_train, y_train)).predict(X_test)
best_predictions = best_clf.predict(X_test)

# Report the before-and-afterscores
print("Unoptimized model\n------")
print("Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions)))
print("Area under curve on testing data: {:.4f}".format(roc_auc_score(y_test, predictions)))
print("\nOptimized Model\n------")
print("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
print("Final Area under curve on  the testing data: {:.4f}".format(roc_auc_score(y_test, best_predictions)))

features_test = pd.read_csv("/Users/snehamitta/Desktop/CharityML/test_census.csv")

# Replace all NaNs with forwardfilling
for row in features_test:
    features_test[row].fillna(method='ffill', axis=0, inplace=True)

#test.isnull().sum()
#test.head()

# Transform Skewed Continuous Features
skewed = ['capital-gain', 'capital-loss']
features_test_log_transformed = pd.DataFrame(data = features_test)
features_test_log_transformed[skewed] = features_test_log_transformed[skewed].apply(lambda x: np.log(x + 1))

#Normalizing Numerical Features
# Initialize a scaler, then apply it to the features
scaler = MinMaxScaler() # default=(0, 1)
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

features_test_log_minmax_transform = pd.DataFrame(data = features_test_log_transformed)
features_test_log_minmax_transform[numerical] = scaler.fit_transform(features_test_log_transformed[numerical])

# One-hot encode the 'features_log_minmax_transform' data using pandas.get_dummies()
features_test_encoded = pd.get_dummies(features_test_log_minmax_transform)

# Remove the first column
features_test_final = features_test_encoded.drop('Unnamed: 0',1)

# Make predictions using features_test_final and store it a new coulmn in test dataset
features_test['id'] = features_test.iloc[:,0]
features_test['income'] = best_clf.predict_proba(features_test_final)[:,1]

# write output file
features_test[['id', 'income']].to_csv("submission.csv", index=False)