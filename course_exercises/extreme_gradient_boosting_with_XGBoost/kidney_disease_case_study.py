# Apply numeric imputer
numeric_imputation_mapper = DataFrameMapper(
    [
        ([numeric_feature], SimpleImputer(strategy="median"))
        for numeric_feature in non_categorical_columns
    ],
    input_df=True,
    df_out=True,
)
