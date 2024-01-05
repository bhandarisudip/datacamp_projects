# Arctic Penguin Exploration: Unraveling Clusters in the Icy Domain with K-means clustering
'''
You have been asked to support a team of researchers who have been collecting data about penguins in Antartica! 

**Origin of this data** : Data were collected and made available by Dr. Kristen Gorman and the Palmer Station, Antarctica LTER, a member of the Long Term Ecological Research Network.

**The dataset consists of 5 columns.**

- culmen_length_mm: culmen length (mm)
- culmen_depth_mm: culmen depth (mm)
- flipper_length_mm: flipper length (mm)
- body_mass_g: body mass (g)
- sex: penguin sex

Unfortunately, they have not been able to record the species of penguin, but they know that there are three species that are native to the region: **Adelie**, **Chinstrap**, and **Gentoo**, so your task is to apply your data science skills to help them identify groups in the dataset!
'''

# Project Instructions
'''
Utilize your unsupervised learning skills to reduce dimensionality and identify clusters in the penguins dataset!

1. Begin by reading in "data/penguins.csv" as a pandas DataFrame called penguins_df, then investigate and clean the dataset by removing the null values and outliers. Save as a cleaned DataFrame called penguins_clean.
2. Pre-process the cleaned data using standard scaling and the one-hot encoding to add dummy variables:
    2.1. Create the dummy variables and remove the original categorical feature from the dataset.
    2.2. Scale the data using the standard scaling method.
    2.3. Save the updated data as a new DataFrame called penguins_preprocessed.
3. Perform Principal Component Analysis (PCA) on the penguins_preprocessed dataset to determine the desired number of components, considering any component with an explained variance ratio above 10% as a suitable component. Save the number of components as a variable called n_components.
    3.1. Finally, execute PCA using n_components and store the result as penguins_PCA.
4. Employ k-means clustering on the penguins_PCA dataset, setting random_state=42, to determine the number of clusters through elbow analysis. Save the optimal number of clusters in a variable called n_cluster.
5. Create and fit a new k-means cluster model, setting n_cluster equal to your n_cluster variable, saving the model as a variable called kmeans.
    5.1. Visualize your clusters using the first two principle components.
6. Add the label column extracted from the k-means clustering (using kmeans.labels_) to the penguins_clean DataFrame.
7. Create a statistical table by grouping penguins_clean based on the "label" column and calculating the mean of each numeric column. Save this table as stat_penguins.
'''

# Import Required Packages
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Loading and examining the dataset
penguins_df = pd.read_csv("penguins.csv")
penguins_df.head()