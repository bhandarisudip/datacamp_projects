# Import Lasso
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

# Instantiate a lasso regression model
lasso = Lasso(alpha=0.3)

# Create X and y arrays
X = sales_df.drop("sales", axis=1).values
y = sales_df["sales"].values

# Fit the model to the data
lasso.fit(X, y)

# Compute and print the coefficients
sales_columns = sales_df.columns
lasso_coef = lasso.fit(X, y).coef_
print(lasso_coef)
plt.bar(sales_columns, lasso_coef)
plt.xticks(rotation=45)
plt.show()
