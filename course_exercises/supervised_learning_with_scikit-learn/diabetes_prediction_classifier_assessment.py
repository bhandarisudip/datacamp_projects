"""
Predict whether or not each individual is likely to have diabetes based 
on the features body mass index (BMI) and age (in years).
"""

#  Import confusion matrix
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Instantiate the KNN model
knn = KNeighborsClassifier(n_neighbors=6)

# Create X and y arrays
X = np.array(
    diabetes_df["diabetes"]
)  # 0 indicates no diabetes, 1 indicate individual has diabetes
y = np.array(diabetes_df[["bmi", "age"]])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Fit the model to the training data
knn.fit(X_train, y_train)

# Predict the labels of the test data: y_pred
y_pred = knn.predict(X_test)

# Generate the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
