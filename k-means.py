import numpy as np
def kmeans(data, k, thresh=1,max_iterations=100):
    center = data[np.random.choice(data.shape[0], k,replace=False)]

    for _ in range(max_iterations):
        distance = np.linalg.norm(data[:, None] - center,axis=2)
        labels = np.argmin(distance, axis=1)
        new_centers = np.array([data[labels == i].mean(axis=0) for i in range(k)])
        if np.all(center == new_centers):
            break
        center_change  = np.linalg.norm(new_centers - center)
        if center_change <thresh:
            break
        center = new_centers
    return labels, center
data = np.random.rand(100,2)
k = 3
labels, center = kmeans(data, k)