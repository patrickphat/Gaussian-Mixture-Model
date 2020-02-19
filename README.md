# Gaussian-Mixture-Model

Gaussian Mixture Model (GMM) is a parametric clustering algorithm that builds up further from K-Means Clustering Algorithm. Instead of representing each cluster with a center of that cluster,  GMM represents each clusters as a gaussian distribution uniquely identified by mean and covariance.

## Usage
### Clustering 
```python
# Settings
max_iters = 25
n_clusters = 3

# Fit the model
model = GaussianMixtureModel(n_clusters=n_clusters)
model.fit(X,max_iters=max_iters)

# Predict classes for clustering
k_pred = model.predict(X)
```

### Anomaly detection
```python

# Using the previous generative model to fit on the data that you want to perform anomaly detection
model.fit_anomaly(X_)

# Predict anomalous instances
anomaly_pred = model.predict_anomaly(X_)
```
