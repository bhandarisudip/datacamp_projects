import matplotlib.pyplot as plt
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    cross_val_score,
    GridSearchCV,
    KFold,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Create features and target variable
X = music_df.drop("popularity", axis=1)
y = music_df["popularity"].values.reshape(-1, 1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Instatntiate StandardScaler
scaler = StandardScaler()

# Scale X_train and X_test
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transfor(X_test)

# Pipeline for predicting song popularity
#  Create models dictionary
models = {
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "Decision Tree Classifier": DecisionTreeClassifier(),
}
results = []

# Loop through the models' values
for model in models.values():

    #  Instantiate a KFold object
    kf = KFold(n_splits=6, random_state=12, shuffle=True)

    # Perform cross-validation
    cv_results = cross_val_score(model, X_train_scaled, y_train, cv=kf)
    results.append(cv_results)
plt.boxplot(results, labels=models.keys())
plt.show()

# Pipeline for predicting song popularity
# Create steps
steps = [
    ("imp_mean", SimpleImputer()),
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression()),
]

# Set up pipeline
pipeline = Pipeline(steps=steps)
params = {
    "logreg__solver": ["newton-cg", "saga", "lbfgs"],
    "logreg__C": np.linspace(0.001, 1.0, 10),
}

# Create the GridSearchCV object
tuning = GridSearchCV(pipeline, param_grid=params)
tuning.fit(X_train, y_train)
y_pred = tuning.predict(X_test)

# Compute and print performance
print(
    "Tuned Logistic Regression Parameters: {}, Accuracy: {}".format(
        tuning.best_params_, tuning.score(X_test, y_test)
    )
)
