# Import necessary libraries
import numpy as np
from scipy.spatial import distance
from sklearn.metrics import precision_score, recall_score, f1_score,rand_score
import matplotlib.pyplot as plt

# Set hyperparameters
alpha = 0.1
n_iterations = 100

# Initialize empty lists for data
X = []
y = []
count = 0

# Loop over each file containing data
for file_name in ['fruits', 'animals', 'countries', 'veggies']:
    # Open the file and loop over each line
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            # Extract the features and label from the line
            parts = line.strip().split()
            X.append(parts[1:301])
            y.append(count)
        count = count + 1

# Convert lists to NumPy arrays and normalize feature values
X = np.array(X).astype(np.float32)
y = np.array(y).astype(np.float32)
X = (X - np.min(X)) / (np.max(X) - np.min(X))

# Print unique labels in the dataset
print(np.unique(y))

# Set range of k values to try
k_values = range(2, 11)

# Initialize empty lists for evaluation metrics
precisions = []
recalls = []
f_scores =[]
random_index = []

# Loop over each value of k
for k in k_values:

    # Add a new column of zeros to X to store cluster labels
    X = np.c_[X, np.zeros(X.shape[0], dtype=int)]
    
    # Initialize weights randomly
    weights = np.random.rand(k, X.shape[1]-1)
    
    # Loop over each iteration of the SOM algorithm
    for iteration in range(n_iterations):

        # Update learning rate
        alpha = alpha * (1 - iteration / n_iterations)
        
        # Shuffle data
        np.random.shuffle(X)
        
        # Compute distances between each data point and each weight vector
        distances = distance.cdist(X[:, :-1], weights)
        closest = np.argmin(distances, axis=1)
        
        # Update weights for each cluster
        for i in range(k):
            mask = closest == i
            if np.any(mask):
                weights[i] += alpha * np.mean(X[mask, :-1] - weights[i], axis=0)
        
    # Assign each data point to the closest cluster
    for i in range(X.shape[0]):
        X[i, -1] = np.argmin(np.apply_along_axis(lambda x: distance.euclidean(x, X[i, :-1].flatten()), 1, weights))

    # Get true and predicted labels
    true_labels = y
    predicted_labels = X[:, -1]
   # Print unique predicted labels
    print(np.unique(predicted_labels))
    
    # Compute evaluation metrics
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f_score = f1_score(true_labels, predicted_labels, average='weighted')
    r_index = rand_score(true_labels,predicted_labels)

    # Append evaluation metrics to lists
    precisions.append(precision)
    recalls.append(recall)
    f_scores.append(f_score)
    random_index.append(r_index)

# Plot evaluation metrics as a function of k
plt.plot(k_values, precisions, label='Precision')
plt.plot(k_values, recalls, label='Recall')
plt.plot(k_values, f_scores, label='F-score')
plt.plot(k_values,random_index,label="rand_index")
plt.xlabel('k')
plt.ylabel('Evaluation metrics')
plt.title('Evaluation metrics vs. k')
plt.legend()
plt.show()

