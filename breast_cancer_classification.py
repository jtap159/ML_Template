import pandas as pd # Import Pandas for data manipulation using dataframes
import numpy as np # Import Numpy for data statistical analysis
import matplotlib.pyplot as plt # Import matplotlib for data visualisation
import seaborn as sns # Statistical data visualization
# Import Cancer data drom the Sklearn library
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns=np.append(cancer['feature_names'], ['target']))
df_cancer.to_csv("breast_cancer_data.csv", index=False)


sns.pairplot(df_cancer, hue='target', vars=['mean radius', 'mean texture', 'mean area', 'mean perimeter', 'mean smoothness'])

sns.countplot(df_cancer['target'], label="Count")
# 212 - Malignant (target=0), 357 - Benign (target=1)

sns.scatterplot(x='mean area', y='mean smoothness', hue='target', data=df_cancer)

sns.heatmap(df_cancer.corr(), annot=True)
