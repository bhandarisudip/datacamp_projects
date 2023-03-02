'''
Make a simple decision tree using scikit-learn's DecisionTreeClassifier. 
Use breast cancer dataset that comes pre-loaded with sklearn to predict tumor as malignant or benign.
'''

# Import the necessary modules
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load the dataset
breast_cancer_data = load_breast_cancer(as_frame=True)
df = breast_cancer_data["data"]
X = df.iloc[:, :-1]
y= df.iloc[:, -1]

# Create the training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Instantiate the classifier: dt_clf_4
dt_clf_4 = DecisionTreeClassifier(max_depth=4)

# Fit the classifier to the training set
dt_clf_4.fit(X_train, y_train)

# Predict the labels of the test set: y_pred_4
y_pred_4 = dt_clf_4.predict(X_test)

# Compute the accuracy of the predictions: accuracy
accuracy = float(np.sum(y_pred_4==y_test))/y_test.shape[0]
print("accuracy:", accuracy)
