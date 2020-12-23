import pandas as pd # Import Pandas for data manipulation using dataframes
import numpy as np # Import Numpy for data statistical analysis
import matplotlib.pyplot as plt # Import matplotlib for data visualisation
import seaborn as sns # Statistical data visualization
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix


# Support Vector Machine (SVM)

# Importing the dataset
dataset = pd.read_csv('breast_cancer_data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Initial data visualization
sns.pairplot(dataset, hue='target', vars=['mean radius', 'mean texture', 'mean area', 'mean perimeter', 'mean smoothness'])

sns.countplot(dataset['target'], label="Count")
# 212 - Malignant (target=0), 357 - Benign (target=1)

sns.scatterplot(x='mean area', y='mean smoothness', hue='target', data=dataset)

sns.heatmap(dataset.corr(), annot=True)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=5)

# Feature Scaling
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

# Feature Scaling
X_train = normalize(X_train, axis=0, norm='max')
X_test = normalize(X_test, axis=0, norm='max')

# Training the SVM model on the Training set
classifier = SVC(kernel='linear', random_state=0)
classifier.fit(X_train, y_train)

# Applying Grid Search to find the best model and the best parameters
parameters = [{'C': [0.25, 0.5, 0.75, 1], 'kernel': ['linear']},
              {'C': [0.25, 0.5, 0.75, 1], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
grid_search = GridSearchCV(estimator=classifier,
                           param_grid=parameters,
                           scoring='accuracy',
                           cv=10,
                           n_jobs=-1)
grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
grid_predict = grid_search.predict(X_test)
cm = confusion_matrix(y_test, grid_predict)
print("Best Parameters:", best_parameters)
print(cm)

# Applying k-Fold Cross Validation to grid search best estimator
accuracies = cross_val_score(estimator=grid_search.best_estimator_, X=X_train, y=y_train, cv=10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))
