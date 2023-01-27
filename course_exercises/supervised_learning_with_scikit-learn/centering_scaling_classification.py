"""
Build a pipeline to scale features in the music_df dataset and perform grid 
search cross-validation using a logistic regression model with different 
values for the hyperparameter C.
"""

# Import necessary modules
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Create feature and targets variable
X = music_df.drop(["Rock"], axis=1).values
y = music_df["Rock"].values.reshape(-1, 1)

# Build the steps for the pipeline

# Scaler object for standardizing the data
scaler = StandardScaler()

# Logistic Regression object
logreg = LogisticRegression()

# Outline the steps
steps = [("scaler", scaler), ("logreg", logreg)]

# Instantiate the Pipeline object
pipeline = Pipeline(steps)

# Create the parameter space for grid search
parameters = {"logreg__C": np.linspace(0.001, 1.0, 20)}

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=21
)

# Instantiate the grid search object
cv = GridSearchCV(pipeline, param_grid=parameters)

# Fit to the training data
cv.fit(X_train, y_train)

# Print the best score and the best parameters
print(cv.best_score_, "\n", cv.best_params_)
