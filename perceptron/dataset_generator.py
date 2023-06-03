import numpy as np
import pandas as pd

# Set the random seed for reproducibility
np.random.seed(42)

# Define the number of samples and features
num_samples = 500
num_features = 2

# Generate random features
X = np.random.uniform(low=0, high=10, size=(num_samples, num_features))

# Generate random labels based on a linear decision boundary
# y = 0 if cgpa + resume_score < 10, y = 1 otherwise
y = np.where(np.sum(X, axis=1) < 10, 0, 1)

# Create a DataFrame to store the dataset
data = pd.DataFrame(X, columns=["cgpa", "resume_score"])
data["placed"] = y

# Save the dataset to a CSV file
data.to_csv("perceptron_dataset.csv", index=False)
