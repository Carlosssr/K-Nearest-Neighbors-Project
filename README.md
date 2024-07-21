K-Nearest Neighbors Project

## Project Overview

This project demonstrates the implementation of the K-Nearest Neighbors (KNN) algorithm using a synthetic dataset. The primary goal is to classify data points based on their features into predefined target classes. This notebook follows a structured approach, including data preprocessing, exploratory data analysis (EDA), model training, evaluation, and optimization.

## Project Structure

The notebook is divided into several sections:

1. **Import Libraries**: Importing essential libraries such as pandas, numpy, matplotlib, seaborn, and scikit-learn.
2. **Get the Data**: Loading the dataset into a pandas DataFrame.
3. **Exploratory Data Analysis (EDA)**: Visualizing the data using seaborn to understand the relationships between features.
4. **Data Preprocessing**: Standardizing the features using `StandardScaler` to ensure all features contribute equally to the distance calculations in the KNN algorithm.
5. **Train-Test Split**: Splitting the data into training and testing sets to evaluate the model's performance.
6. **Model Training**: Training a KNN classifier with the training data.
7. **Model Evaluation**: Evaluating the model's performance using a confusion matrix and classification report.
8. **Hyperparameter Tuning**: Using the elbow method to determine the optimal number of neighbors (K) for the KNN classifier.
9. **Retraining and Final Evaluation**: Retraining the model with the optimal K value and evaluating its performance.

## How to Use This Notebook

1. **Clone or Download the Repository**: Ensure you have the notebook file (`03-K Nearest Neighbors Project - Solutions.ipynb`) and the dataset file (`KNN_Project_Data.csv`).
2. **Install Necessary Libraries**: Make sure you have the required libraries installed. You can install them using pip:
    ```sh
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```
3. **Run the Notebook**: Open the notebook in Jupyter Notebook or any compatible environment and run the cells sequentially to follow the steps and reproduce the results.

## Key Steps and Code Snippets

### Importing Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
```

### Loading the Data
```python
df = pd.read_csv('KNN_Project_Data.csv')
df.head()
```

### Exploratory Data Analysis
```python
sns.pairplot(df, hue='TARGET CLASS', palette='coolwarm')
```

### Data Preprocessing
```python
scaler = StandardScaler()
scaler.fit(df.drop('TARGET CLASS', axis=1))
scaled_features = scaler.transform(df.drop('TARGET CLASS', axis=1))
df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])
df_feat.head()
```

### Train-Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(scaled_features, df['TARGET CLASS'], test_size=0.30)
```

### Model Training and Evaluation
```python
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
```

### Hyperparameter Tuning
```python
error_rate = []
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10, 6))
plt.plot(range(1, 40), error_rate, color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
```

### Retraining with Optimal K Value
```python
knn = KNeighborsClassifier(n_neighbors=30)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)
print('WITH K=30')
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
```

## Conclusion

This project demonstrates the implementation and evaluation of the K-Nearest Neighbors algorithm. By following the steps outlined in this notebook, you will gain insights into data preprocessing, model training, and hyperparameter tuning using the KNN algorithm.
