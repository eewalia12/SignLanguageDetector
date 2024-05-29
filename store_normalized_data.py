import pandas as pd
import numpy as np

def normalize_landmarks(df):
    landmarks = df.iloc[:, 1:].values.reshape(-1, 21, 3)
    wrist_coords = landmarks[:, 0, :].reshape(-1, 1, 3)
    normalized_landmarks = landmarks - wrist_coords

    max_dist = np.linalg.norm(normalized_landmarks, axis=2).max(axis=1).reshape(-1, 1, 1)  # Shape: (n_samples, 1, 1)
    normalized_landmarks /= max_dist  # Scale coordinates
    return normalized_landmarks.reshape(-1, 63)  # Flatten the array back to (n_samples, 63)

# Load your original data
df = pd.read_csv('./outputAdditional.csv')

# Normalize the landmarks
normalized_data = normalize_landmarks(df)

# Create a DataFrame from the normalized data
columns=[f'landmark_{i}_{axis}' for i in range(21) for axis in ['x', 'y', 'z']]
normalized_df = pd.DataFrame(normalized_data, columns=columns)

# Optionally, add the first column (e.g., IDs) back to the normalized dataframe
normalized_df.insert(0, df.columns[0], df.iloc[:, 0])

# Save the normalized data to a new CSV file
normalized_df.to_csv('normalized_additionaldata.csv', index=False)