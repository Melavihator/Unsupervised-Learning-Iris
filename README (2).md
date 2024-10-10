# Unsupervised learning on Iris Dataset

## K-Means and Agglomerative Clustering on the Iris Data

### Project Overview

This project implements K-Means and Agglomerative Clustering algorithms on the Iris dataset. The objective is to cluster iris flowers based on their morphological features and to visualize the resulting clusters. This project provides insights into the effectiveness of both clustering techniques for the Iris dataset.

## Dataset

The Iris dataset is a well-known dataset in the field of machine learning and statistics, consisting of 150 samples of iris flowers. Each sample includes four features:

- **Sepal Length** (cm)
- **Sepal Width** (cm)
- **Petal Length** (cm)
- **Petal Width** (cm)

Each sample is categorized into one of three species:
- **Iris Setosa**
- **Iris Versicolor**
- **Iris Virginica**

## Project Structure

```
kmeans_agglomerative_clustering/
├── README.md              # Project documentation
├── kmeans_clustering.py    # Script for K-Means clustering
└── agglomerative_clustering.py # Script for Agglomerative Clustering

```

## Installation

To run this project, you need Python installed on your system. Additionally, the following libraries are required:

- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- SciPy

You can install the required libraries using pip. If you have a `requirements.txt` file, you can install all dependencies at once:

```bash
pip install -r requirements.txt
```

### Example `requirements.txt`
```
numpy
pandas
matplotlib
scikit-learn
scipy
```

## Usage

### K-Means Clustering

1. **Load the Dataset**: Load the Iris dataset.
2. **Preprocess the Data**: Standardize the feature values.
3. **Apply K-Means Clustering**: Use the KMeans class from Scikit-learn to cluster the data.
4. **Visualize the Clusters**: Plot the clusters with different colors.

#### Example Code for K-Means Clustering

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('iris_data.csv')

# Prepare the feature matrix
x = data.iloc[:, :-1].values  # All features except the last column

# Standardize the features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
y_kmeans = kmeans.fit_predict(x_scaled)

# Visualize the clusters
plt.figure(figsize=(10, 6))
unique_labels = np.unique(y_kmeans)
for label in unique_labels:
    plt.scatter(x_scaled[y_kmeans == label, 0], x_scaled[y_kmeans == label, 1], label=f'Cluster {label}')
plt.title('K-Means Clustering on Iris Dataset')
plt.xlabel('Feature 1 (Standardized)')
plt.ylabel('Feature 2 (Standardized)')
plt.legend()
plt.show()
```

### Agglomerative Clustering

1. **Load the Dataset**: Load the Iris dataset.
2. **Preprocess the Data**: Standardize the feature values.
3. **Apply Agglomerative Clustering**: Use the AgglomerativeClustering class from Scikit-learn.
4. **Visualize the Dendrogram**: Create a dendrogram to show the hierarchical clustering.

#### Example Code for Agglomerative Clustering

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Load the dataset
data = pd.read_csv('iris_data.csv')

# Prepare the feature matrix
x = data.iloc[:, :-1].values  # All features except the last column

# Standardize the features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Perform Agglomerative Clustering
linkage_matrix = linkage(x_scaled, method='ward')

# Create the dendrogram
plt.figure(figsize=(10, 8))
dendrogram(linkage_matrix, labels=data['species'].values, leaf_rotation=30)
plt.title("Dendrogram for Agglomerative Clustering on Iris Dataset")
plt.xlabel("Species")
plt.ylabel('Euclidean distances')
plt.show()
```

## Results

- **K-Means Clustering**: The K-Means algorithm effectively clusters the Iris dataset into three distinct clusters, which can be visualized in a scatter plot. Each cluster is represented by a different color, demonstrating the separation between species based on their features.

- **Agglomerative Clustering**: The dendrogram visualizes the hierarchical relationships among the Iris species. The height of the branches indicates the distance between merged clusters, which can be used to determine the appropriate number of clusters.

## Conclusion

This project demonstrates the application of K-Means and Agglomerative Clustering techniques to the Iris dataset, highlighting the effectiveness of these algorithms for clustering tasks. The visualizations provide valuable insights into the relationships among the species and their morphological characteristics.

## License

This project is licensed under the MIT License. See the LICENSE file for more information.

## Acknowledgments

- UCI Machine Learning Repository for providing the Iris dataset.
- The Scikit-learn and SciPy documentation for their excellent resources on clustering algorithms.
```

### Instructions for Customization
- Update any paths to files or scripts as necessary.
- Include any specific acknowledgments or references for datasets, libraries, or tutorials you utilized in your work.
- You can also add a **"Future Work"** section if you plan to extend the project further or include additional analysis.



 