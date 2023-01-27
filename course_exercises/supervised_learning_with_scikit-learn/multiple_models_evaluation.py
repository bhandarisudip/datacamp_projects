"""
Build and visualize the results of three different models to classify whether 
a song is popular or not."""

# Import the necessary modules
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import (
    cross_val_score,
    GridSearchCV,
    KFold,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler

# Convert popularity to a binary feature
music_df["popularity"] = np.where(
    music_df["popularity"] >= np.median(music_df["popularity"]), 1, 0
)

# Create a dictionary instantiating the three models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=0.1),
    "Lasso": Lasso(alpha=0.1),
}

# Create empty list to store the model results
results = []

# Create features and target variable
X = music_df.drop("popular", axis=1)
y = music_df["popular"]

# Split features and target variable using train-test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Instatntiate StandardScaler
scaler = StandardScaler()

# Scale X_train and X_test
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transfor(X_test)

# Loop through the models' values
for model in models.values():

    #  Instantiate a KFold object
    kf = KFold(n_splits=6, random_state=12, shuffle=True)

    # Perform cross-validation
    cv_results = cross_val_score(model, X_train_scaled, y_train, cv=kf)
    results.append(cv_results)

# Create a box plot of the results
plt.boxplot(results, labels=models.keys())
plt.show()

# Predicting on the test set

# Import mean_squared_error
for name, model in models.items():

    # Fit the model to the training data
    model.fit(X_train_scaled, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test_scaled)

    # Calculate the test_rmse
    test_rmse = mean_squared_error(y_test, y_pred, squared=False)
    print("{} Test Set RMSE: {}".format(name, test_rmse))
