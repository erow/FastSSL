
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
import torch

def evaluate_clustering(assignments, true_labels):
    """
    Evaluates the clustering performance using Normalized Mutual Information (NMI).

    Parameters:
    assignments (np.ndarray): The cluster assignments for each data point, shape (num_points,).
    true_labels (np.ndarray): The true labels for each data point, shape (num_points,).

    Returns:
    nmi (float): The Normalized Mutual Information score.
    """
    if isinstance(assignments, torch.Tensor):
        assignments = assignments.cpu().numpy()
    if isinstance(true_labels, torch.Tensor):
        true_labels = true_labels.cpu().numpy()
    
    nmi = normalized_mutual_info_score(true_labels, assignments)
    return nmi

class StreamingKMeans:
    def __init__(self, k, lr=0.01):
        """
        Initialize Streaming K-Means.

        Args:
            k (int): Number of clusters.
            dim (int): Dimensionality of the data.
            lr (float): Learning rate for updating cluster centers.
        """
        self.k = k
        self.lr = lr
        self.cluster_centers = None  # Initialize cluster centers randomly

    def update_predict(self, data):
        """
        Update cluster centers with new data.

        Args:
            data (torch.Tensor): A tensor of shape (batch_size, dim) containing the new data points.
        """
        batch_size = data.size(0)
        
        
        if self.cluster_centers is None:
            # Initialize cluster centers with the first batch of data
            self.dim = data.size(1)
            self.cluster_centers =  torch.randn(self.k, self.dim).to(data.device)
        
        # Compute distances between data points and cluster centers
        distances = torch.cdist(data, self.cluster_centers)  # Shape: (batch_size, k)
        
        # Assign each data point to the nearest cluster
        nearest_cluster_indices = torch.argmin(distances, dim=1)  # Shape: (batch_size,)
        
        # Update cluster centers
        for i in range(self.k):
            # Get the data points assigned to the i-th cluster
            cluster_data = data[nearest_cluster_indices == i]
            
            if cluster_data.size(0) > 0:
                # Compute the mean of the data points in the cluster
                cluster_mean = torch.mean(cluster_data, dim=0)
                
                # Update the cluster center using the learning rate
                self.cluster_centers[i] = (1 - self.lr) * self.cluster_centers[i] + self.lr * cluster_mean
        return nearest_cluster_indices

    def predict(self, data):
        """
        Predict the nearest cluster for each data point.

        Args:
            data (torch.Tensor): A tensor of shape (batch_size, dim) containing the data points.

        Returns:
            torch.Tensor: A tensor of shape (batch_size,) containing the cluster indices.
        """
        distances = torch.cdist(data, self.cluster_centers)
        return torch.argmin(distances, dim=1)

class KmeansProb():
    def __init__(self, representation_fn, num_clusters=100):
        self.representation_fn = representation_fn
        self.num_clusters = num_clusters
        self.kmeans_dict = {}
        
    def step(self,x,y):
        if isinstance(x,list):
            x = x[0]
        log = self.step_train(x.cuda(), y.cuda())
        return log
    
    def step_train(self, x,y):
        log = {}
        if isinstance(x, list) or isinstance(x, tuple):
            x = x[0]
        for name, z in self.representation_fn(x).items():
            if name not in self.kmeans_dict:
                self.kmeans_dict[name] = StreamingKMeans(self.num_clusters)
            assignments = self.kmeans_dict[name].update_predict(z)
            nmi = evaluate_clustering(assignments, y)
            log[name+'@nmi'] = nmi
        return log
    
# Example usage
if __name__ == "__main__":
    # Generate some random data
    np.random.seed(42)
    num_samples = 20000
    noise = np.random.rand(num_samples, 128).astype(np.float32)
    true_labels = np.random.randint(0, 10, num_samples)  # Assuming we have 10 true clusters
    true_centroids = np.random.rand(10, 20).astype(np.float32)
    W = np.random.rand(128, 20).astype(np.float32)/4
    
    # for k in np.linspace(0,2,10):
    #     data = noise*k + true_centroids[true_labels].dot(W.T)

    #     # Run k-means clustering
    #     num_clusters = 10
    #     centroids, assignments = run_kmeans(data, num_clusters)

    #     # Evaluate clustering performance
    #     nmi = evaluate_clustering(assignments, true_labels)
    #     print(": Normalized Mutual Information (NMI):", nmi, "k:", k)
    
    skm = StreamingKmeans(num_clusters=100)
    data = noise*0.1 + true_centroids[true_labels].dot(W.T)
    for chunk,labels in zip(np.array_split(data, 10), np.array_split(true_labels, 10)):
        assignments = skm.push(chunk)
        nmi = evaluate_clustering(assignments, labels) if assignments is not None else None
        print("Streaming Kmeans: Normalized Mutual Information (NMI):", nmi)
