'''
Run cross validation on churn_data.

Note: when using xgboost's scikit-learn API, the input datasets 
are converted into DMatrix data on the fly, but when one uses
the xgboost cv object, you have to first explicitly convert
one's data into a DMatrix. So, that's what we'll will do here 
before running cross-validation on churn_data.
'''

# Import the necessary modules
import xgboost as xgb

# Create arrays for the features and the target: X, y
X, y = churn_data.iloc[:,:-1], churn_data.iloc[:,-1]

# Create the DMatrix from X and y: churn_dmatrix
churn_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary: params
params = {"objective":"reg:logistic", "max_depth":3}

# Perform cross-validation: cv_results
cv_results = xgb.cv(
    dtrain=churn_dmatrix,
    params=params,
    nfold=3,
    num_boost_round=5,
    metrics="error",
    as_pandas=True,
    seed=123
    )

# Print cv_results
print(cv_results)

# Print the accuracy
print(((1-cv_results["test-error-mean"]).iloc[-1]))
