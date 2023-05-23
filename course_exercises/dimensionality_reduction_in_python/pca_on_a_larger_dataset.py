from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Scale the data
scaler = StandardScaler()
ansur_std = scaler.fit_transform(ansur_df)

# Apply PCA
pca = PCA()
foo = pca.fit(ansur_std)

# Inspect the explained variance ratio per component
print(pca.explained_variance_ratio_)