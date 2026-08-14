# ============================================================
# WEEK 3 - IRIS FLOWER CLUSTERING PROJECT
# Machine Learning & AI Internship
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, adjusted_rand_score

import joblib


# ============================================================
# 1. PROJECT SETTINGS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)


print("=" * 60)
print("WEEK 3 - IRIS FLOWER CLUSTERING PROJECT")
print("=" * 60)


# ============================================================
# 2. LOAD IRIS DATASET
# ============================================================

print("\n========== LOADING IRIS DATASET ==========")

iris = load_iris()

X = iris.data
y = iris.target

feature_names = iris.feature_names
target_names = iris.target_names


print("Dataset shape:", X.shape)

print("\nFeature names:")
print(feature_names)

print("\nTarget names:")
print(target_names)


# ============================================================
# 3. CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(
    X,
    columns=feature_names
)

df["Target"] = y

print("\n========== FIRST 5 ROWS ==========")
print(df.head())


# ============================================================
# 4. DATA INFORMATION
# ============================================================

print("\n========== DATA INFORMATION ==========")

print("\nDataset shape:")
print(df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nStatistics:")
print(df.describe())


# ============================================================
# 5. FEATURE SCALING
# ============================================================

print("\n========== FEATURE SCALING ==========")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("Original shape:", X.shape)
print("Scaled shape  :", X_scaled.shape)

print("\nScaled data - first 5 rows:")
print(X_scaled[:5])


# ============================================================
# 6. ELBOW METHOD
# ============================================================

print("\n========== ELBOW METHOD ==========")

inertia_values = []

K_range = range(1, 11)

for k in K_range:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    inertia_values.append(model.inertia_)

print("\nK values:")
print(list(K_range))

print("\nInertia values:")
print(inertia_values)


# Save Elbow Plot

plt.figure(figsize=(8, 5))

plt.plot(
    list(K_range),
    inertia_values,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Choosing K")
plt.grid(True)

elbow_file = os.path.join(
    OUTPUT_DIR,
    "elbow_method.png"
)

plt.savefig(elbow_file)

plt.close()

print("\nElbow plot saved to:")
print(elbow_file)


# ============================================================
# 7. K-MEANS CLUSTERING
# ============================================================

print("\n========== K-MEANS CLUSTERING ==========")

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)

print("Number of clusters:", 3)

print("\nCluster labels:")
print(np.unique(clusters))

print("\nCluster centers:")
print(kmeans.cluster_centers_)


# ============================================================
# 8. ADD CLUSTERS TO DATAFRAME
# ============================================================

df["Cluster"] = clusters

print("\n========== CLUSTERED DATA ==========")

print(df.head(10))


# ============================================================
# 9. CLUSTER COUNTS
# ============================================================

print("\n========== CLUSTER COUNTS ==========")

cluster_counts = df["Cluster"].value_counts().sort_index()

print(cluster_counts)


# ============================================================
# 10. VISUALIZE CLUSTERS
# ============================================================

print("\n========== CLUSTER VISUALIZATION ==========")

plt.figure(figsize=(8, 6))

plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=clusters
)

plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])

plt.title("Iris K-Means Clusters")

plt.grid(True)

cluster_file = os.path.join(
    OUTPUT_DIR,
    "kmeans_clusters.png"
)

plt.savefig(cluster_file)

plt.close()

print("Cluster visualization saved to:")
print(cluster_file)


# ============================================================
# 11. COMPARE PREDICTED CLUSTERS WITH TRUE LABELS
# ============================================================

print("\n========== CLUSTER VS TRUE LABELS ==========")

comparison_df = pd.DataFrame({
    "True_Label": y,
    "Predicted_Cluster": clusters
})

print(comparison_df.head(20))


# ============================================================
# 12. CONFUSION MATRIX
# ============================================================

print("\n========== CONFUSION MATRIX ==========")

cm = confusion_matrix(
    y,
    clusters
)

print(cm)

cm_file = os.path.join(
    OUTPUT_DIR,
    "cluster_confusion_matrix.csv"
)

pd.DataFrame(cm).to_csv(
    cm_file,
    index=False
)

print("\nConfusion matrix saved to:")
print(cm_file)


# ============================================================
# 13. ADJUSTED RAND INDEX
# ============================================================

print("\n========== CLUSTERING SCORE ==========")

ari_score = adjusted_rand_score(
    y,
    clusters
)

print("Adjusted Rand Index:", ari_score)


# ============================================================
# 14. PCA - DIMENSIONALITY REDUCTION
# ============================================================

print("\n========== PCA DIMENSIONALITY REDUCTION ==========")

pca = PCA(
    n_components=2
)

X_pca = pca.fit_transform(X_scaled)

print("Original dimensions:", X_scaled.shape[1])

print("Reduced dimensions:", X_pca.shape[1])


# ============================================================
# 15. EXPLAINED VARIANCE RATIO
# ============================================================

print("\n========== EXPLAINED VARIANCE RATIO ==========")

explained_variance = pca.explained_variance_ratio_

print("PC1:", explained_variance[0])

print("PC2:", explained_variance[1])

print(
    "Total explained variance:",
    explained_variance.sum()
)


# ============================================================
# 16. PCA VISUALIZATION
# ============================================================

print("\n========== PCA VISUALIZATION ==========")

plt.figure(figsize=(8, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=clusters
)

plt.xlabel("Principal Component 1")

plt.ylabel("Principal Component 2")

plt.title("Iris Clusters After PCA")

plt.grid(True)

pca_file = os.path.join(
    OUTPUT_DIR,
    "pca_clusters.png"
)

plt.savefig(pca_file)

plt.close()

print("PCA plot saved to:")
print(pca_file)


# ============================================================
# 17. TRUE LABEL VISUALIZATION
# ============================================================

print("\n========== TRUE LABEL VISUALIZATION ==========")

plt.figure(figsize=(8, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=y
)

plt.xlabel("Principal Component 1")

plt.ylabel("Principal Component 2")

plt.title("Iris True Labels After PCA")

plt.grid(True)

true_label_file = os.path.join(
    OUTPUT_DIR,
    "pca_true_labels.png"
)

plt.savefig(true_label_file)

plt.close()

print("True label plot saved to:")
print(true_label_file)


# ============================================================
# 18. SAVE CLUSTERED DATA
# ============================================================

print("\n========== SAVING CLUSTERED DATA ==========")

clustered_file = os.path.join(
    OUTPUT_DIR,
    "iris_clustered.csv"
)

df.to_csv(
    clustered_file,
    index=False
)

print("Clustered dataset saved to:")
print(clustered_file)


# ============================================================
# 19. SAVE PCA DATA
# ============================================================

pca_df = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "True_Label": y,
    "Cluster": clusters
})

pca_csv_file = os.path.join(
    OUTPUT_DIR,
    "iris_pca.csv"
)

pca_df.to_csv(
    pca_csv_file,
    index=False
)

print("\nPCA dataset saved to:")
print(pca_csv_file)


# ============================================================
# 20. SAVE K-MEANS MODEL
# ============================================================

print("\n========== SAVING MODEL ==========")

model_file = os.path.join(
    OUTPUT_DIR,
    "iris_kmeans_model.joblib"
)

joblib.dump(
    kmeans,
    model_file
)

print("K-Means model saved to:")
print(model_file)


# ============================================================
# 21. SAVE SCALER
# ============================================================

scaler_file = os.path.join(
    OUTPUT_DIR,
    "iris_scaler.joblib"
)

joblib.dump(
    scaler,
    scaler_file
)

print("Scaler saved to:")
print(scaler_file)


# ============================================================
# 22. LOAD MODEL
# ============================================================

print("\n========== LOADING SAVED MODEL ==========")

loaded_model = joblib.load(
    model_file
)

loaded_scaler = joblib.load(
    scaler_file
)

loaded_predictions = loaded_model.predict(
    loaded_scaler.transform(X)
)

print("Model loaded successfully.")

print(
    "Predictions from loaded model:",
    loaded_predictions[:10]
)


# ============================================================
# 23. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("WEEK 3 PROJECT SUMMARY")
print("=" * 60)

print("\nDataset:")
print("Iris Dataset")

print("\nSamples:")
print(len(X))

print("\nFeatures:")
print(X.shape[1])

print("\nK-Means clusters:")
print(3)

print("\nAdjusted Rand Index:")
print(ari_score)

print("\nPCA components:")
print(2)

print("\nExplained variance:")
print(explained_variance)

print("\nTotal explained variance:")
print(explained_variance.sum())


# ============================================================
# 24. FINAL FILES
# ============================================================

print("\n========== FILES CREATED ==========")

print("1.", elbow_file)
print("2.", cluster_file)
print("3.", cm_file)
print("4.", pca_file)
print("5.", true_label_file)
print("6.", clustered_file)
print("7.", pca_csv_file)
print("8.", model_file)
print("9.", scaler_file)

print("\n" + "=" * 60)
print("WEEK 3 IRIS CLUSTERING COMPLETED SUCCESSFULLY")
print("=" * 60)