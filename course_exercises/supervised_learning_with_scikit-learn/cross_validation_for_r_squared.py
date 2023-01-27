"""
Build a linear regression model, then use 6-fold cross-validation to assess
its accuracy for predicting sales using social media advertising expenditure.
"""

# Import the necessary modules
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score

# Create X from the social media column's values
X = np.array(sales_df["radio"])

# Reshape X
X = X.reshape(-1, 1)

# Create y from the sales column's values
y = np.array(sales_df["sales"])

#  Create a KFold object
kf = KFold(n_splits=6, shuffle=True, random_state=5)

reg = LinearRegression()

# Compute 6-fold cross-validation scores
cv_scores = cross_val_score(reg, X, y, cv=kf)

# Print scores
print(cv_scores)

# Analyze cross-validation metrics

# Print the mean
print(np.mean(cv_scores))

# Print the standard deviation
print(np.std(cv_scores))

# Print the 95% confidence interval
print(np.quantile(cv_scores, [0.025, 0.975]))
