# Week 3 - Iris Flower Clustering Project

## Machine Learning & AI Internship

This project implements unsupervised learning using the Iris dataset.

## Project Objective

The objective of this project is to:

- Apply K-Means clustering
- Use K = 3 clusters
- Apply feature scaling
- Use the Elbow Method
- Visualize clusters
- Compare predicted clusters with true labels
- Apply PCA for dimensionality reduction
- Calculate explained variance ratio
- Save and load the trained K-Means model

## Dataset

The project uses the Iris dataset available through Scikit-learn.

Dataset details:

- Samples: 150
- Features: 4
- Classes: 3

Features:

1. Sepal Length
2. Sepal Width
3. Petal Length
4. Petal Width

Classes:

- Setosa
- Versicolor
- Virginica

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Joblib

## Machine Learning Techniques

### 1. Feature Scaling

StandardScaler is used to standardize the four numerical features.

### 2. Elbow Method

The Elbow Method is used to evaluate different values of K.

K values from 1 to 10 were tested.

### 3. K-Means Clustering

K-Means clustering is applied with:

K = 3

### 4. PCA

Principal Component Analysis is used to reduce the dataset from 4 dimensions to 2 dimensions.

### 5. Model Evaluation

Adjusted Rand Index is used to compare clustering results with the true labels.

## Results

### K-Means

Number of clusters:

3

Cluster sizes:

- Cluster 0: 53
- Cluster 1: 50
- Cluster 2: 47

### Adjusted Rand Index

0.6201

### PCA Explained Variance

PC1: 72.96%

PC2: 22.85%

Total explained variance: 95.81%

## Generated Outputs

The following files are generated inside the `outputs` folder:

- elbow_method.png
- kmeans_clusters.png
- cluster_confusion_matrix.csv
- pca_clusters.png
- pca_true_labels.png
- iris_clustered.csv
- iris_pca.csv
- iris_kmeans_model.joblib
- iris_scaler.joblib

## How to Run

Create and activate the environment:

```bash
conda create -n week3_ml python=3.11 -y
conda activate week3_ml