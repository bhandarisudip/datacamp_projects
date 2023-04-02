'''
Use the Comic Con dataset to: 
- define cluster centers through kmeans() function.
- assign cluster labels through the vq() function.
'''

# Import the kmeans and vq functions
from scipy.cluster.vq import kmeans, vq

# Generate cluster centers
cluster_centers, distortion = kmeans(comic_con[["x_scaled", "y_scaled"]], k_or_guess=2)

# Assign cluster labels
comic_con['cluster_labels'], distortion_list = vq(comic_con[["x_scaled", "y_scaled"]], cluster_centers)

# Plot clusters
sns.scatterplot(x='x_scaled', y='y_scaled', 
                hue='cluster_labels', data = comic_con)
plt.show()
