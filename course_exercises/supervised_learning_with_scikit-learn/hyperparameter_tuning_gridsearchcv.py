# Import GridSearchCV
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold, GridSearchCV, train_test_split

# Create X and y arrays
X = diabetes_df.drop("glucose", axis=1).values
y = diabetes_df["glucose"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

#  Set up the parameter grid
param_grid = {"alpha": np.linspace(0.00001, 1, 20)}

# Instanstiate Lasso regression model
lasso = Lasso()

# Instantiate lasso_cv
lasso_cv = GridSearchCV(lasso, param_grid=param_grid, cv=kf)

# Fit to the training data
lasso_cv.fit(X_train, y_train)
print("Tuned lasso paramaters: {}".format(lasso_cv.best_params_))
print("Tuned lasso score: {}".format(lasso_cv.best_score_))
